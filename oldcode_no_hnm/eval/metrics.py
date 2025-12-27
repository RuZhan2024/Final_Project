#!/usr/bin/env python3
"""Compute threshold sweeps + operating-point metrics for a window dataset.

It evaluates a trained model checkpoint on a directory of window .npz files
(e.g. .../test) and produces a JSON report containing:

- AP / ROC-AUC (window-level)
- A threshold sweep with Precision / Recall / F1 / FPR
- An (estimated) FA per 24h curve derived from window-level false-alarm *events*
  grouped by video_id
- Operating-point metrics if an ops YAML is provided

Important about FA/24h
---------------------
To compute FA/24h we need an estimate of total observed time.
We support two methods:

1) If --pose_npz_dir is provided, we load the *sequence* npz for each video_id
   to get the true number of frames (recommended).
2) Otherwise we approximate duration per video as
      (max(w_end) - min(w_start) + 1) / fps
   This can be biased if your window set is *sampled* (e.g. balanced sampling).

The output JSON includes meta.fa24h_method so you can see which method was used.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Small utils
# -----------------------------


def _list_npz_files(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    return [f for f in files if os.path.isfile(f)]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def p_fall_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Works for (B,1) sigmoid logits or (B,2) softmax logits.
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).squeeze(-1)
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


# -----------------------------
# YAML I/O (tiny)
# -----------------------------


def yaml_load_simple(path: str) -> dict:
    """Load a tiny YAML subset used for ops files.

    Supports:
      key: value
      nested:
        key: value

    Values parsed as float/int/bool/str.
    """

    def parse_scalar(s: str):
        s = s.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", s) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+", s
        ):
            return float(s)
        return s

    out: dict = {}
    stack: List[Tuple[int, dict]] = [(0, out)]

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            line = raw.strip("\n")
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            while stack and indent < stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]

            if val == "":
                nxt: dict = {}
                cur[key] = nxt
                stack.append((indent + 2, nxt))
            else:
                cur[key] = parse_scalar(val)

    return out


def yaml_dump_simple(obj: dict, path: str) -> None:
    """Write simple YAML (dicts + scalars)."""

    def dump_value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return "null"
            return str(v)
        return json.dumps(str(v))  # quote strings safely

    lines: List[str] = []

    def rec(d: dict, indent: int):
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                rec(v, indent + 2)
            else:
                lines.append(" " * indent + f"{k}: {dump_value(v)}")

    rec(obj, 0)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------
# Window feature builders
# -----------------------------


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32)
    return x


def _get_mask_from_npz(z: np.lib.npyio.NpzFile, conf_gate: float, use_precomputed_mask: bool) -> Optional[np.ndarray]:
    """Return (T,V) mask in {0,1} where 1 means valid joint."""
    if use_precomputed_mask and "mask" in z.files:
        m = z["mask"].astype(np.float32)
        if m.ndim == 3:
            # sometimes stored as (T,V,1)
            m = m[..., 0]
        return m

    # If no precomputed mask, derive from confidence if present.
    if "joints" in z.files:
        joints = z["joints"]
        if joints.shape[-1] >= 3:
            conf = joints[..., 2]
            return (conf >= conf_gate).astype(np.float32)
    return None


def read_window_tcn(
    npz_path: str,
    *,
    center: str = "pelvis",
    use_motion: bool = True,
    use_conf_channel: bool = True,
    motion_scale_by_fps: bool = True,
    conf_gate: float = 0.20,
    use_precomputed_mask: bool = True,
) -> np.ndarray:
    """Build TCN input X of shape (T, C)."""
    z = np.load(npz_path, allow_pickle=False)

    if "joints" not in z.files:
        raise KeyError(f"Missing 'joints' in {npz_path}")

    joints = _ensure_float32(z["joints"])  # (T,V,2/3)
    T, V, D = joints.shape

    fps = float(z["fps"]) if "fps" in z.files else 30.0

    # Optional centering (pelvis ~= joint 23/24 midpoint for MediaPipe, but your pipeline may store pelvis already)
    if center == "pelvis":
        # If you used pelvis normalization in preprocess, this is often already centered.
        # Here we keep a conservative approach: no additional centering unless explicit pelvis coords exist.
        pass

    xy = joints[..., :2]  # (T,V,2)

    feats: List[np.ndarray] = [xy.reshape(T, V * 2)]

    if use_motion:
        if "motion" in z.files:
            motion = _ensure_float32(z["motion"])[..., :2]
        else:
            motion = np.zeros_like(xy)
            motion[1:] = xy[1:] - xy[:-1]
        if motion_scale_by_fps and fps > 0:
            motion = motion * fps
        feats.append(motion.reshape(T, V * 2))

    if use_conf_channel:
        if D >= 3:
            conf = joints[..., 2]
        else:
            conf = np.ones((T, V), dtype=np.float32)
        feats.append(conf.reshape(T, V))

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # Apply mask if available
    mask = _get_mask_from_npz(z, conf_gate=conf_gate, use_precomputed_mask=use_precomputed_mask)
    if mask is not None:
        # mask is (T,V) -> expand to xy/motion channels; keep conf channel as-is.
        mask2 = np.repeat(mask, 2, axis=1)  # (T, V*2)
        cursor = 0
        # xy
        X[:, cursor : cursor + V * 2] *= mask2
        cursor += V * 2
        if use_motion:
            X[:, cursor : cursor + V * 2] *= mask2
            cursor += V * 2
        # conf channel not masked

    return X


def read_window_gcn(
    npz_path: str,
    *,
    center: str = "pelvis",
    use_motion: bool = True,
    use_conf_channel: bool = True,
    motion_scale_by_fps: bool = True,
    conf_gate: float = 0.20,
    use_precomputed_mask: bool = True,
) -> np.ndarray:
    """Build GCN input X of shape (T, V, F)."""
    z = np.load(npz_path, allow_pickle=False)

    joints = _ensure_float32(z["joints"])  # (T,V,2/3)
    T, V, D = joints.shape
    fps = float(z["fps"]) if "fps" in z.files else 30.0

    xy = joints[..., :2]
    feats: List[np.ndarray] = [xy]

    if use_motion:
        if "motion" in z.files:
            motion = _ensure_float32(z["motion"])[..., :2]
        else:
            motion = np.zeros_like(xy)
            motion[1:] = xy[1:] - xy[:-1]
        if motion_scale_by_fps and fps > 0:
            motion = motion * fps
        feats.append(motion)

    if use_conf_channel:
        if D >= 3:
            conf = joints[..., 2:3]
        else:
            conf = np.ones((T, V, 1), dtype=np.float32)
        feats.append(conf.astype(np.float32))

    X = np.concatenate(feats, axis=-1).astype(np.float32)  # (T,V,F)

    mask = _get_mask_from_npz(z, conf_gate=conf_gate, use_precomputed_mask=use_precomputed_mask)
    if mask is not None:
        # mask (T,V) -> (T,V,1)
        X[..., :2] *= mask[..., None]
        if use_motion:
            X[..., 2:4] *= mask[..., None]

    return X


# -----------------------------
# Models (TCN + GCN, compatible with our training scripts)
# -----------------------------


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        y = self.drop(self.act(self.bn1(self.conv1(x))))
        y = self.drop(self.act(self.bn2(self.conv2(y))))
        return self.act(y + self.skip(x))


class SimpleTCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_ch, hid, k=3, dropout=dropout),
            TCNBlock(hid, hid, k=3, dropout=dropout),
            TCNBlock(hid, hid, k=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        y = self.net(x)
        return self.head(y)

class TCNBlockV2(nn.Module):
    """
    Matches state_dict keys:
      blocks.{i}.conv.*
      blocks.{i}.bn.*
    """
    def __init__(self, hid: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(hid, hid, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(self.conv(x)))
        y = self.drop(y)
        return F.relu(x + y)


class SimpleTCNv2(nn.Module):
    """
    Matches state_dict keys:
      conv_in.0.*   (Conv1d)
      conv_in.1.*   (BatchNorm1d)
      blocks.{i}.*
      head.*
    """
    def __init__(
        self,
        in_ch: int,
        hid: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        k_in: int = 5,
        out_dim: int = 1,
    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=k_in, padding=k_in // 2),
            nn.BatchNorm1d(hid),
        )
        self.blocks = nn.ModuleList([TCNBlockV2(hid, kernel=3, dropout=dropout) for _ in range(depth)])
        self.head = nn.Linear(hid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.relu(self.conv_in(x))
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=-1)  # global average over time
        return self.head(x)


def _detect_tcn_variant(sd: Dict[str, torch.Tensor]) -> str:
    keys = sd.keys()
    if any(k.startswith("conv_in.") for k in keys):
        return "v2"
    if any(k.startswith("net.") for k in keys):
        return "legacy"
    return "legacy"


def _infer_tcn_v2_params(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    w = sd["conv_in.0.weight"]  # (hid, in_ch, k_in)
    in_ch = int(w.shape[1])
    hid = int(w.shape[0])
    k_in = int(w.shape[2])

    block_ids = set()
    for k in sd.keys():
        m = re.match(r"blocks\.(\d+)\.conv\.weight$", k)
        if m:
            block_ids.add(int(m.group(1)))
    depth = (max(block_ids) + 1) if block_ids else 0

    out_dim = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 1

    return {"in_ch": in_ch, "hid": hid, "depth": depth, "k_in": k_in, "out_dim": out_dim}

# ---- GCN blocks (based on train_gcn_rewritten.py)


def _build_mediapipe_edges() -> List[Tuple[int, int]]:
    # MediaPipe Pose has 33 landmarks. This edge list is a pragmatic subset.
    # It doesn't need to be perfect; consistency matters more.
    edges = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
    ]
    # Make undirected
    out = set()
    for a, b in edges:
        out.add((a, b))
        out.add((b, a))
    return sorted(out)


def _adjacency_matrix(num_nodes: int = 33) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes):
        A[i, i] = 1.0
    for a, b in _build_mediapipe_edges():
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            A[a, b] = 1.0
    # Normalize (D^-1 A)
    deg = A.sum(dim=1).clamp(min=1.0)
    A = A / deg[:, None]
    return A


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(1, ch // r))
        self.fc2 = nn.Linear(max(1, ch // r), ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,V)
        s = x.mean(dim=(2, 3))
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)
        return x * s


class GraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.lin = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,V)
        # graph aggregation on V: (B,C,T,V) @ (V,V)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        return self.lin(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        self.gc = GraphConv(in_ch, out_ch, A)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.tc = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SE(out_ch) if use_se else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,V)
        y = self.act(self.bn1(self.gc(x)))
        y = self.drop(self.act(self.bn2(self.tc(y))))
        y = self.se(y)
        return self.act(y + self.skip(x))


class GraphTCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        num_nodes: int = 33,
    ):
        super().__init__()
        A = _adjacency_matrix(num_nodes)
        self.register_buffer("A", A)

        self.stgcn = nn.Sequential(
            STGCNBlock(in_feats, gcn_hidden, A, dropout=dropout, use_se=use_se),
            STGCNBlock(gcn_hidden, gcn_hidden, A, dropout=dropout, use_se=use_se),
            STGCNBlock(gcn_hidden, gcn_hidden, A, dropout=dropout, use_se=use_se),
        )

        self.tcn = nn.Sequential(
            nn.Conv1d(gcn_hidden * num_nodes, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_hidden, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,F)
        B, T, V, F = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,F,T,V)
        y = self.stgcn(x)  # (B,C,T,V)
        y = y.permute(0, 2, 3, 1).contiguous().view(B, T, V * y.shape[1])  # (B,T,V*C)
        y = y.transpose(1, 2)  # (B,V*C,T)
        y = self.tcn(y)  # (B,tcn_hidden,T)
        return self.head(y)


class TwoStreamGCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        fuse: str = "concat",
        num_nodes: int = 33,
    ):
        super().__init__()
        self.fuse = fuse

        # stream A: joints only
        self.stream_a = GraphTCN(
            in_feats=in_feats,
            gcn_hidden=gcn_hidden,
            tcn_hidden=tcn_hidden,
            dropout=dropout,
            use_se=use_se,
            num_nodes=num_nodes,
        )
        # stream B: same input, but different parameters (acts as a second view)
        self.stream_b = GraphTCN(
            in_feats=in_feats,
            gcn_hidden=gcn_hidden,
            tcn_hidden=tcn_hidden,
            dropout=dropout,
            use_se=use_se,
            num_nodes=num_nodes,
        )

        out_dim = 2 if fuse == "concat" else 1
        self.fuse_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.stream_a(x)  # (B,1)
        b = self.stream_b(x)  # (B,1)
        if self.fuse == "mean":
            y = 0.5 * (a + b)
            return y
        # concat
        y = torch.cat([a, b], dim=1)
        return self.fuse_head(y)



# ---- Two-stream GCN (enc_j/enc_m) to match train_gcn.py checkpoints ----

class SEBlock(nn.Module):
    """Squeeze-and-Excitation over channel dimension for [B,T,V,C]."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(4, ch // r)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, v, c = x.shape
        z = x.mean(dim=(1, 2))  # [B,C]
        z = self.relu(self.fc1(z))
        s = self.sig(self.fc2(z)).view(b, 1, 1, c)
        return x * s


class GCNBlockLN(nn.Module):
    """Graph conv (A_hat aggregation) + Linear + LayerNorm + ReLU + Dropout."""
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.35):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(out_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,Cin]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)
        x = self.lin(x)
        x = self.drop(self.relu(self.ln(x)))
        return x


class GCNTemporalEncoder(nn.Module):
    """Encoder used in train_gcn.py: outputs an embedding vector [B,E]."""
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
        # We load A_hat from checkpoint buffer, but define a placeholder here.
        A = torch.eye(num_joints, dtype=torch.float32)
        self.register_buffer("A_hat", A)

        self.g1 = GCNBlockLN(in_feats, gcn_hidden, dropout=dropout)
        self.g2 = GCNBlockLN(gcn_hidden, gcn_hidden, dropout=dropout)

        self.use_se = bool(use_se)
        self.se = SEBlock(gcn_hidden) if self.use_se else nn.Identity()

        self.temporal = nn.Sequential(
            nn.Conv1d(gcn_hidden, tcn_hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_hidden, tcn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.g1(x, self.A_hat)
        h = self.g2(h, self.A_hat)
        h = self.se(h)
        h = h.mean(dim=2)        # [B,T,C]
        h = h.permute(0, 2, 1)   # [B,C,T]
        h = self.temporal(h).squeeze(-1)  # [B,E]
        return h


class TwoStreamGCNEnc(nn.Module):
    """Two-stream model matching checkpoints with keys enc_j.*, enc_m.*, head.*"""
    def __init__(
        self,
        num_joints: int,
        in_feats_joint: int,
        in_feats_motion: int,
        gcn_hidden: int = 96,
        tcn_hidden: int = 192,
        dropout: float = 0.35,
        use_se: bool = True,
        fuse: str = "concat",  # concat|sum
    ):
        super().__init__()
        self.enc_j = GCNTemporalEncoder(num_joints, in_feats_joint, gcn_hidden, tcn_hidden, dropout, use_se)
        self.enc_m = GCNTemporalEncoder(num_joints, in_feats_motion, gcn_hidden, tcn_hidden, dropout, use_se)
        self.fuse = str(fuse)

        if self.fuse == "sum":
            self.head = nn.Linear(tcn_hidden, 1)
        else:
            self.head = nn.Linear(tcn_hidden * 2, 1)

    def forward(self, xj: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        ej = self.enc_j(xj)
        em = self.enc_m(xm)
        if self.fuse == "sum":
            e = 0.5 * (ej + em)
        else:
            e = torch.cat([ej, em], dim=-1)
        return self.head(e)


def _is_two_stream_enc_ckpt(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("enc_j.") for k in sd.keys()) and any(k.startswith("enc_m.") for k in sd.keys())


def _infer_two_stream_enc_params(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    # Required tensors
    Aj = sd.get("enc_j.A_hat", None)
    if Aj is None or not isinstance(Aj, torch.Tensor) or Aj.ndim != 2:
        raise RuntimeError("GCN ckpt looks like two-stream, but missing enc_j.A_hat")

    wj = sd.get("enc_j.g1.lin.weight", None)
    wm = sd.get("enc_m.g1.lin.weight", None)
    if wj is None or wm is None:
        raise RuntimeError("GCN ckpt missing enc_j.g1.lin.weight / enc_m.g1.lin.weight")

    num_joints = int(Aj.shape[0])
    in_feats_joint = int(wj.shape[1])
    in_feats_motion = int(wm.shape[1])
    gcn_hidden = int(wj.shape[0])

    # tcn_hidden from temporal conv
    tw = sd.get("enc_j.temporal.0.weight", None)
    if tw is None or not isinstance(tw, torch.Tensor) or tw.ndim != 3:
        raise RuntimeError("GCN ckpt missing enc_j.temporal.0.weight to infer tcn_hidden")
    tcn_hidden = int(tw.shape[0])

    use_se = "enc_j.se.fc1.weight" in sd

    # fuse from head weight width
    hw = sd.get("head.weight", None)
    if hw is None or not isinstance(hw, torch.Tensor) or hw.ndim != 2:
        raise RuntimeError("GCN ckpt missing head.weight")
    if int(hw.shape[1]) == tcn_hidden:
        fuse = "sum"
    elif int(hw.shape[1]) == 2 * tcn_hidden:
        fuse = "concat"
    else:
        # fallback: assume concat if divisible by 2
        fuse = "concat" if int(hw.shape[1]) == 2 * tcn_hidden else "concat"

    return {
        "num_joints": num_joints,
        "in_feats_joint": in_feats_joint,
        "in_feats_motion": in_feats_motion,
        "gcn_hidden": gcn_hidden,
        "tcn_hidden": tcn_hidden,
        "use_se": use_se,
        "fuse": fuse,
    }


def _split_packed_gcn_two_stream(
    xb: torch.Tensor,
    *,
    use_motion: bool,
    use_conf_channel: bool,
    expected_joint: Optional[int] = None,
    expected_motion: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split packed features (xy, motion, conf) into xj and xm streams.

    xb: [B,T,V,Ffull] where
      - xy   at [:,:, :, 0:2]
      - motion at [:,:, :, 2:4]   (requires use_motion=1)
      - conf at [:,:, :, 4:5]     (requires use_conf_channel=1)
    """
    if not use_motion:
        raise ValueError("Two-stream GCN requires use_motion=1 (checkpoint expects motion stream).")

    Ffull = int(xb.shape[-1])
    if use_conf_channel:
        if Ffull < 5:
            raise ValueError(f"Expected packed gcn features with conf channel (F>=5), got F={Ffull}.")
        conf = xb[..., 4:5]
        xj = torch.cat([xb[..., 0:2], conf], dim=-1)
        xm = torch.cat([xb[..., 2:4], conf], dim=-1)
    else:
        if Ffull < 4:
            raise ValueError(f"Expected packed gcn features with motion (F>=4), got F={Ffull}.")
        xj = xb[..., 0:2]
        xm = xb[..., 2:4]

    if expected_joint is not None and int(xj.shape[-1]) != int(expected_joint):
        raise ValueError(f"Joint-stream feature dim mismatch: got {int(xj.shape[-1])}, ckpt expects {int(expected_joint)}.")
    if expected_motion is not None and int(xm.shape[-1]) != int(expected_motion):
        raise ValueError(f"Motion-stream feature dim mismatch: got {int(xm.shape[-1])}, ckpt expects {int(expected_motion)}.")
    return xj, xm


# -----------------------------
# Checkpoint/model loader
# -----------------------------


def get_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    # Some users save raw state_dict directly.
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    raise ValueError("Cannot find model state_dict in checkpoint")

_get_state_dict = get_state_dict

def _infer_tcn_in_ch(sd: dict) -> Optional[int]:
    for k, v in sd.items():
        if k.endswith("conv1.weight") and isinstance(v, torch.Tensor) and v.ndim == 3:
            return int(v.shape[1])
    # fallback: any Conv1d weight
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 3 and "weight" in k:
            return int(v.shape[1])
    return None


def build_model_from_ckpt(arch: str, ckpt_path: str, device: torch.device, strict: Optional[bool] = None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {}) or {}
    sd = get_state_dict(ckpt)

    if strict is None:
        strict = True

    if arch == "tcn":
        variant = _detect_tcn_variant(sd)

        if variant == "v2":
            p = _infer_tcn_v2_params(sd)
            # cfg can override inferred values if you saved them
            model = SimpleTCNv2(
                in_ch=int(cfg.get("in_ch", p["in_ch"])),
                hid=int(cfg.get("hid", p["hid"])),
                depth=int(cfg.get("depth", p["depth"])),
                dropout=float(cfg.get("dropout", 0.0)),
                k_in=int(cfg.get("k_in", p["k_in"])),
                out_dim=int(cfg.get("out_dim", p["out_dim"])),
            )
            cfg = {**cfg, "tcn_variant": "v2"}

        else:
            # legacy path (your existing SimpleTCN)
            in_ch = _infer_tcn_in_ch(sd, fallback=cfg.get("in_ch", 66))
            hid = int(cfg.get("hid", 128))
            depth = int(cfg.get("depth", 3))
            dropout = float(cfg.get("dropout", 0.0))
            out_dim = int(cfg.get("out_dim", 1))
            model = SimpleTCN(in_ch=in_ch, hid=hid, depth=depth, dropout=dropout, out_dim=out_dim)
            cfg = {**cfg, "tcn_variant": "legacy"}
    elif arch == "gcn":
        # Support two families of GCN checkpoints:
        #  - train_gcn.py two-stream encoders (enc_j.*, enc_m.*, head.*)
        #  - older graph-tcn style (optional / best-effort)
        if _is_two_stream_enc_ckpt(sd):
            p = _infer_two_stream_enc_params(sd)
            model = TwoStreamGCNEnc(
                num_joints=int(cfg.get("num_joints", p["num_joints"])),
                in_feats_joint=int(cfg.get("in_feats_joint", p["in_feats_joint"])),
                in_feats_motion=int(cfg.get("in_feats_motion", p["in_feats_motion"])),
                gcn_hidden=int(cfg.get("gcn_hidden", p["gcn_hidden"])),
                tcn_hidden=int(cfg.get("tcn_hidden", p["tcn_hidden"])),
                dropout=float(cfg.get("dropout", 0.0)),
                use_se=bool(cfg.get("use_se", p["use_se"])),
                fuse=str(cfg.get("fuse", p["fuse"])),
            )
            cfg = {
                **cfg,
                "gcn_variant": "two_stream_enc",
                "num_joints": int(p["num_joints"]),
                "in_feats_joint": int(p["in_feats_joint"]),
                "in_feats_motion": int(p["in_feats_motion"]),
                "tcn_hidden": int(p["tcn_hidden"]),
                "fuse": str(p["fuse"]),
            }
        else:
            # Fallback: try to load the simple GraphTCN / TwoStreamGCN defined above
            # (only works for older checkpoints that match those class names/keys).
            if any(k.startswith("stream_a.") for k in sd.keys()) and any(k.startswith("stream_b.") for k in sd.keys()):
                # infer in_feats from first conv weight in stream_a
                w = None
                for k, v in sd.items():
                    if k.endswith("stream_a.stgcn.0.gc.lin.weight") and isinstance(v, torch.Tensor) and v.ndim == 4:
                        w = v
                        break
                in_feats = int(w.shape[1]) if w is not None else int(cfg.get("in_feats", 5))
                model = TwoStreamGCN(
                    in_feats=in_feats,
                    gcn_hidden=int(cfg.get("gcn_hidden", 96)),
                    tcn_hidden=int(cfg.get("tcn_hidden", 192)),
                    dropout=float(cfg.get("dropout", 0.0)),
                    use_se=bool(cfg.get("use_se", True)),
                    fuse=str(cfg.get("fuse", "concat")),
                    num_nodes=int(cfg.get("num_nodes", 33)),
                )
                cfg = {**cfg, "gcn_variant": "two_stream_legacy", "in_feats": in_feats}
            else:
                raise RuntimeError(
                    "Unsupported GCN checkpoint format for eval/metrics.py. "
                    "Expected keys like enc_j.* / enc_m.* (two-stream) or stream_a.* / stream_b.* (legacy)."
                )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.to(device)
    model.load_state_dict(sd, strict=bool(strict))
    model.eval()
    return model, cfg


# -----------------------------
# Metrics / sweeps
# -----------------------------


def _confusion(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32)
    y_hat = y_hat.astype(np.int32)
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    return tp, fp, tn, fn


def sweep_on_unique_thresholds(y_true: np.ndarray, p: np.ndarray) -> dict:
    # Use unique thresholds in descending order (including 1.0 sentinel).
    p = np.asarray(p, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    thr = np.unique(p)
    thr = np.concatenate([thr, np.array([1.0], dtype=np.float64)])
    thr = np.unique(thr)
    thr = np.sort(thr)[::-1]

    prec, rec, f1, fpr = [], [], [], []

    for t in thr:
        y_hat = (p >= t).astype(np.int32)
        tp, fp, tn, fn = _confusion(y_true, y_hat)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1s = 2 * precision * recall / (precision + recall + 1e-12)
        fprs = fp / (fp + tn + 1e-12)
        prec.append(float(precision))
        rec.append(float(recall))
        f1.append(float(f1s))
        fpr.append(float(fprs))

    return {
        "thr": thr.tolist(),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
    }


def count_events_from_flags(flags: Sequence[bool]) -> int:
    # Counts 0->1 transitions.
    c = 0
    prev = False
    for f in flags:
        f = bool(f)
        if f and not prev:
            c += 1
        prev = f
    return c


@dataclass
class WindowMeta:
    video_id: str
    w_start: int
    w_end: int
    fps: float


class WindowDataset(Dataset):
    def __init__(
        self,
        root: str,
        arch: str,
        *,
        center: str = "pelvis",
        use_motion: bool = True,
        use_conf_channel: bool = True,
        motion_scale_by_fps: bool = True,
        conf_gate: float = 0.20,
        use_precomputed_mask: bool = True,
    ):
        self.files = _list_npz_files(root)
        self.arch = arch
        self.center = center
        self.use_motion = use_motion
        self.use_conf_channel = use_conf_channel
        self.motion_scale_by_fps = motion_scale_by_fps
        self.conf_gate = conf_gate
        self.use_precomputed_mask = use_precomputed_mask

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        z = np.load(path, allow_pickle=False)
        y = int(z["y"]) if "y" in z.files else 0
        video_id = str(z["video_id"]) if "video_id" in z.files else os.path.basename(path).split("___")[0]
        w_start = int(z["w_start"]) if "w_start" in z.files else 0
        w_end = int(z["w_end"]) if "w_end" in z.files else w_start
        fps = float(z["fps"]) if "fps" in z.files else 30.0

        if self.arch == "tcn":
            X = read_window_tcn(
                path,
                center=self.center,
                use_motion=self.use_motion,
                use_conf_channel=self.use_conf_channel,
                motion_scale_by_fps=self.motion_scale_by_fps,
                conf_gate=self.conf_gate,
                use_precomputed_mask=self.use_precomputed_mask,
            )
            x = torch.from_numpy(X)  # (T,C)
        else:
            X = read_window_gcn(
                path,
                center=self.center,
                use_motion=self.use_motion,
                use_conf_channel=self.use_conf_channel,
                motion_scale_by_fps=self.motion_scale_by_fps,
                conf_gate=self.conf_gate,
                use_precomputed_mask=self.use_precomputed_mask,
            )
            x = torch.from_numpy(X)  # (T,V,F)

        meta = WindowMeta(video_id=video_id, w_start=w_start, w_end=w_end, fps=fps)
        return x, torch.tensor([y], dtype=torch.float32), meta


def _collate(batch):
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y, list(metas)


def _infer_dataset_name(eval_dir: str) -> str:
    parts = re.split(r"[\\/]+", os.path.normpath(eval_dir))
    # often .../processed/<dataset>/windows_.../(val|test)
    for i, p in enumerate(parts):
        if p == "processed" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: use parent name
    return parts[-2] if len(parts) >= 2 else parts[-1]


def _build_pose_len_map(pose_npz_dir: str) -> Dict[str, int]:
    """Map stem -> length (frames)."""
    mapping: Dict[str, int] = {}
    for p in glob.glob(os.path.join(pose_npz_dir, "**", "*.npz"), recursive=True):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            z = np.load(p, allow_pickle=False)
            if "joints" in z.files:
                mapping[stem] = int(z["joints"].shape[0])
        except Exception:
            continue
    return mapping


def _estimate_video_duration_sec(metas: List[WindowMeta], pose_len_map: Optional[Dict[str, int]] = None) -> float:
    if not metas:
        return 0.0
    vid = metas[0].video_id
    fps = float(metas[0].fps) if metas[0].fps > 0 else 30.0

    if pose_len_map is not None and vid in pose_len_map:
        return float(pose_len_map[vid]) / fps

    starts = [m.w_start for m in metas]
    ends = [m.w_end for m in metas]
    return float(max(ends) - min(starts) + 1) / fps


def _event_fa24h_and_event_recall(
    y_true: np.ndarray,
    p: np.ndarray,
    metas: List[WindowMeta],
    thr: float,
    *,
    pose_len_map: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, float]:
    """Compute (fa24h, recall_event, hours_total).

    - fa24h is based on counting contiguous runs of FP windows per video.
    - recall_event treats each *video* that contains any positive windows as one "event".
    """
    # group indices by video
    by_vid: Dict[str, List[int]] = {}
    for i, m in enumerate(metas):
        by_vid.setdefault(m.video_id, []).append(i)

    false_events = 0
    tp_evt = 0
    fn_evt = 0
    total_sec = 0.0

    for vid, idxs in by_vid.items():
        # sort by time
        idxs = sorted(idxs, key=lambda i: metas[i].w_start)
        ms = [metas[i] for i in idxs]
        total_sec += _estimate_video_duration_sec(ms, pose_len_map)

        yv = y_true[idxs]
        pv = p[idxs]
        pos_flags = (pv >= thr)

        # FP windows are predicted-positive windows where y==0
        fp_flags = pos_flags & (yv == 0)
        false_events += count_events_from_flags(fp_flags.tolist())

        # Event recall: a "positive event" exists if any y==1 windows in this video
        has_event = bool((yv == 1).any())
        if has_event:
            detected = bool((pos_flags & (yv == 1)).any())
            if detected:
                tp_evt += 1
            else:
                fn_evt += 1

    hours = total_sec / 3600.0
    fa24h = float(false_events) / (hours + 1e-12) * 24.0
    recall_event = float(tp_evt) / (tp_evt + fn_evt + 1e-12)
    return fa24h, recall_event, hours


def _evaluate_at_threshold(y_true: np.ndarray, p: np.ndarray, metas: List[WindowMeta], thr: float, pose_len_map=None) -> dict:
    y_hat = (p >= thr).astype(np.int32)
    tp, fp, tn, fn = _confusion(y_true, y_hat)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)

    fa24h, recall_evt, hours = _event_fa24h_and_event_recall(y_true, p, metas, thr, pose_len_map=pose_len_map)

    return {
        "thr": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fa24h": float(fa24h),
        "recall_event": float(recall_evt),
        "hours_total": float(hours),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=None, help="Alias for --fps_default")

    ap.add_argument("--arch", required=True, choices=["tcn", "gcn"])
    ap.add_argument("--eval_dir", required=True, help="Directory with window .npz files (e.g. .../test)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ops", default="", help="Path to ops YAML (optional)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=256)

    # feature options (should match training)
    ap.add_argument("--center", default="pelvis", choices=["pelvis", "none"])
    ap.add_argument("--use_motion", type=int, default=1)
    ap.add_argument("--use_conf_channel", type=int, default=1)
    ap.add_argument("--motion_scale_by_fps", type=int, default=1)
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", type=int, default=1)

    ap.add_argument("--pose_npz_dir", default="", help="Optional: processed pose_npz dir to estimate true durations")
    ap.add_argument("--dataset_name", default="", help="Optional: name for the report")

    args = ap.parse_args()
    if args.fps is not None:
        args.fps_default = args.fps

    device = pick_device()
    model, cfg = build_model_from_ckpt(args.arch, args.ckpt, device)

    pose_len_map = None
    fa24h_method = "window_range"
    if args.pose_npz_dir:
        pose_len_map = _build_pose_len_map(args.pose_npz_dir)
        fa24h_method = "pose_npz"

    ds = WindowDataset(
        args.eval_dir,
        args.arch,
        center=args.center,
        use_motion=bool(args.use_motion),
        use_conf_channel=bool(args.use_conf_channel),
        motion_scale_by_fps=bool(args.motion_scale_by_fps),
        conf_gate=float(args.conf_gate),
        use_precomputed_mask=bool(args.use_precomputed_mask),
    )

    if len(ds) == 0:
        raise SystemExit(f"No .npz windows found in: {args.eval_dir}")

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=_collate)

    y_true_list: List[int] = []
    p_list: List[float] = []
    metas_all: List[WindowMeta] = []

    with torch.no_grad():
        for xb, yb, metas in dl:
            xb = xb.to(device)
            if args.arch == "gcn" and cfg.get("gcn_variant") == "two_stream_enc":
                xj, xm = _split_packed_gcn_two_stream(
                    xb,
                    use_motion=bool(args.use_motion),
                    use_conf_channel=bool(args.use_conf_channel),
                    expected_joint=cfg.get("in_feats_joint"),
                    expected_motion=cfg.get("in_feats_motion"),
                )
                logits = model(xj, xm)
            else:
                logits = model(xb)
            probs = p_fall_from_logits(logits).detach().cpu().numpy().astype(np.float64)
            y_true_list.extend(yb.detach().cpu().numpy().astype(np.int32).tolist())
            p_list.extend(probs.tolist())
            metas_all.extend(metas)

    y_true = np.asarray(y_true_list, dtype=np.int32)
    p = np.asarray(p_list, dtype=np.float64)

    # AP/AUC
    ap_val = None
    auc_val = None
    if average_precision_score is not None:
        try:
            ap_val = float(average_precision_score(y_true, p))
        except Exception:
            ap_val = None
    if roc_auc_score is not None:
        try:
            auc_val = float(roc_auc_score(y_true, p))
        except Exception:
            auc_val = None

    sweep = sweep_on_unique_thresholds(y_true, p)

    # add FA/24h curve
    fa24h_curve = []
    recall_evt_curve = []
    hours_curve = []
    for t in sweep["thr"]:
        fa24h, rec_evt, hours = _event_fa24h_and_event_recall(y_true, p, metas_all, float(t), pose_len_map=pose_len_map)
        fa24h_curve.append(float(fa24h))
        recall_evt_curve.append(float(rec_evt))
        hours_curve.append(float(hours))

    sweep["fa24h"] = fa24h_curve
    sweep["recall_event"] = recall_evt_curve
    sweep["hours_total"] = hours_curve

    # operating points
    ops_report = {}
    ops_cfg = {}
    if args.ops:
        ops_cfg = yaml_load_simple(args.ops)
        for name, v in ops_cfg.items():
            if isinstance(v, dict) and "thr" in v:
                ops_report[name] = _evaluate_at_threshold(y_true, p, metas_all, float(v["thr"]), pose_len_map=pose_len_map)

    dataset = args.dataset_name or _infer_dataset_name(args.eval_dir)

    out = {
        "dataset": dataset,
        "arch": args.arch,
        "eval_dir": args.eval_dir,
        "ckpt": args.ckpt,
        "n_windows": int(len(y_true)),
        "class_balance": {"pos": int((y_true == 1).sum()), "neg": int((y_true == 0).sum())},
        "ap": ap_val,
        "auc": auc_val,
        "sweep": sweep,
        "ops": ops_report,
        "meta": {
            "fa24h_method": fa24h_method,
            "pose_npz_dir": args.pose_npz_dir or None,
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # small console summary
    best_idx = int(np.argmax(np.asarray(sweep["f1"], dtype=np.float64)))
    print(
        f"[OK] report -> {args.out_json} | windows={len(y_true)} pos={out['class_balance']['pos']} neg={out['class_balance']['neg']}"
    )
    print(
        f"[best F1] thr={sweep['thr'][best_idx]:.3f} f1={sweep['f1'][best_idx]:.3f} P={sweep['precision'][best_idx]:.3f} R={sweep['recall'][best_idx]:.3f} fa24h={sweep['fa24h'][best_idx]:.2f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
