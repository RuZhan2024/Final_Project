#!/usr/bin/env python3
"""
fit_ops.py — Fit Operating Points (thresholds) on a validation split.

This script:
  1) Loads a trained checkpoint (TCN or GCN).
  2) Runs inference on a validation windows directory.
  3) Sweeps thresholds to build a recall / precision / FA-per-24h curve.
  4) Picks three operating points (OP1 high-recall, OP2 balanced, OP3 low-alarm).
  5) Writes a small YAML file consumed by eval/metrics.py.

Why this exists:
  - Our project refactored the TCN architecture (checkpoint keys changed).
  - Our windows format evolved (npz may contain either:
        (a) prebuilt 'x' + 'y'  [legacy]
        (b) 'xy' + 'conf' (+ optional 'mask', 'fps')  [current]
    )
  - We want eval + fit to share the same preprocessing knobs.

Author: project helper (self-contained; no eval_common.py dependency).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except Exception:
    yaml = None


# -------------------------
# Reproducibility
# -------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# YAML helper
# -------------------------

def yaml_dump_simple(obj: dict, out_path: str) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)



# -------------------------
# Path parsing helpers
# -------------------------
_WINDOWS_RE = re.compile(r"windows_W(?P<W>\d+)_S(?P<S>\d+)")

def _parse_stride_from_dir(path: str) -> Optional[int]:
    """Extract stride S from a windows directory path like .../windows_W48_S12/val."""
    m = _WINDOWS_RE.search(str(path))
    if not m:
        return None
    try:
        return int(m.group("S"))
    except Exception:
        return None

def _parse_window_from_dir(path: str) -> Optional[int]:
    """Extract window length W from a windows directory path like .../windows_W48_S12/val."""
    m = _WINDOWS_RE.search(str(path))
    if not m:
        return None
    try:
        return int(m.group("W"))
    except Exception:
        return None

# -------------------------
# Models (match train_*.py)
# -------------------------
# We keep a legacy SimpleTCN for backward compatibility (older checkpoints),
# and the current TCN used by train_tcn.py (conv_in / blocks / head).

class SimpleTCN(nn.Module):
    """Legacy TCN (older checkpoints)."""
    def __init__(self, in_ch: int, hid: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hid, hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.mean(dim=-1)  # GAP over time
        return self.head(x).squeeze(-1)


class ResidualBlock1D(nn.Module):
    """Current residual block used by TCN (matches checkpoint keys blocks.*.conv/bn)."""
    def __init__(self, ch: int, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class TCN(nn.Module):
    """Current TCN used in training (matches checkpoint keys conv_in.*, blocks.*, head.*)."""
    def __init__(self, in_ch: int, hidden: int = 128, num_blocks: int = 4, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock1D(hidden, kernel=kernel, dropout=dropout) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        for blk in self.blocks:
            x = blk(x)
        # Global average pooling over time -> (B, hidden)
        x = x.mean(dim=-1)
        # Linear head expects (B, hidden)
        return self.head(x).squeeze(-1)


# ---- GCN (train_gcn.py) ----

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} (A+I) D^{-1/2} normalization."""
    A = A.astype(np.float32)
    I = np.eye(A.shape[0], dtype=np.float32)
    A_hat = A + I
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    """
    A small adjacency for MediaPipe Pose 33 landmarks.
    We use a reasonable subset of edges (undirected). This must match training.
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation over channel dimension."""
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
        z = x.mean(dim=(1, 2))     # [B,C]
        z = self.relu(self.fc1(z))
        s = self.sig(self.fc2(z)).view(b, 1, 1, c)
        return x * s


class GCNBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.30):
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
    """
    Produces an embedding vector per sample.
      input : [B,T,V,F]
      output: [B,E]
    """
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
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))

        self.g1 = GCNBlock(in_feats, gcn_hidden, dropout=dropout)
        self.g2 = GCNBlock(gcn_hidden, gcn_hidden, dropout=dropout)

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
        h = h.mean(dim=2)        # pool joints -> [B,T,F]
        h = h.permute(0, 2, 1)   # [B,F,T]
        h = self.temporal(h).squeeze(-1)  # [B,E]
        return h


class TwoStreamGCN(nn.Module):
    """
    Two-stream encoder:
      stream A: joints(+conf)
      stream B: motion(+conf)

    We fuse embeddings by concat -> linear head.
    """
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

def build_model_from_ckpt(ckpt_path: str, arch: str, device: torch.device, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    cfg = ckpt.get("cfg", {}) or {}

    if arch == "tcn":
        in_ch = ckpt.get("in_ch", None)
        if in_ch is None:
            in_ch = _infer_in_ch_from_state_dict(sd)
        if in_ch is None:
            raise RuntimeError("Cannot infer TCN input channels from checkpoint.")

        hid = int(cfg.get("hid", 128))
        dropout = float(cfg.get("dropout", 0.3))
        num_blocks = int(cfg.get("num_blocks", 4))
        kernel = int(cfg.get("kernel", 3))

        is_new = any(k.startswith("conv_in.") or k.startswith("blocks.") or k.startswith("head.") for k in sd.keys())

        tried = []
        for use_new in ([is_new, not is_new] if isinstance(is_new, bool) else [True, False]):
            try:
                model = (TCN(in_ch, hid, num_blocks, kernel, dropout) if use_new else SimpleTCN(in_ch, hid, dropout)).to(device)
                model.load_state_dict(sd, strict=strict)
                model.eval()
                return model, cfg, {"in_ch": in_ch}
            except Exception as e:
                tried.append((use_new, str(e)))
                continue
        raise RuntimeError("Failed to load TCN ckpt. Tried: " + "; ".join([f"new={u}: {msg}" for u, msg in tried]))

    if arch == "gcn":
        num_joints = int(ckpt.get("num_joints", cfg.get("num_joints", 33)))
        in_feats_full = int(ckpt.get("in_feats", cfg.get("in_feats", 2)))  # per-joint feature dim F_full
        gcn_hid = int(cfg.get("gcn_hidden", 96))
        tcn_hid = int(cfg.get("tcn_hidden", 192))
        dropout = float(cfg.get("dropout", 0.35))
        use_se = bool(cfg.get("use_se", True))

        # Detect two-stream checkpoints (enc_j/enc_m) vs single-stream.
        two_stream = bool(cfg.get("two_stream", False)) or any(k.startswith("enc_j.") for k in sd.keys())
        fuse = str(cfg.get("fuse", "concat"))

        # Infer fuse from head weight shape when possible.
        if "head.weight" in sd:
            hw = int(sd["head.weight"].shape[1])
            if hw == tcn_hid:
                fuse = "sum"
            elif hw == tcn_hid * 2:
                fuse = "concat"

        if two_stream:
            has_conf = (in_feats_full == 5)
            in_j = 3 if has_conf else 2
            in_m = 3 if has_conf else 2
            model = TwoStreamGCN(
                num_joints=num_joints,
                in_feats_joint=in_j,
                in_feats_motion=in_m,
                gcn_hidden=gcn_hid,
                tcn_hidden=tcn_hid,
                dropout=dropout,
                use_se=use_se,
                fuse=fuse,
            ).to(device)
            model.load_state_dict(sd, strict=strict)
            model.eval()
            return model, cfg, {"num_joints": num_joints, "in_feats": in_feats_full, "two_stream": True, "fuse": fuse}

        # Single-stream: encoder + linear head (trained as nn.Sequential in train_gcn.py)
        model = nn.Sequential(
            GCNTemporalEncoder(
                num_joints=num_joints,
                in_feats=in_feats_full,
                gcn_hidden=gcn_hid,
                tcn_hidden=tcn_hid,
                dropout=dropout,
                use_se=use_se,
            ),
            nn.Linear(tcn_hid, 1),
        ).to(device)

        model.load_state_dict(sd, strict=False if not strict else True)
        model.eval()
        return model, cfg, {"num_joints": num_joints, "in_feats": in_feats_full, "two_stream": False}

    raise ValueError(f"Unknown arch: {arch}")


def pelvis_center(xy: np.ndarray, conf: np.ndarray, gate: float = 0.2) -> np.ndarray:
    """
    Compute pelvis center (midpoint of hips) with confidence gating.
    MediaPipe hips: left=23, right=24.
    """
    lh, rh = 23, 24
    lh_ok = conf[:, lh] >= gate
    rh_ok = conf[:, rh] >= gate

    c = np.zeros((xy.shape[0], 2), dtype=np.float32)
    both = lh_ok & rh_ok
    c[both] = 0.5 * (xy[both, lh] + xy[both, rh])

    only_l = lh_ok & (~rh_ok)
    c[only_l] = xy[only_l, lh]

    only_r = rh_ok & (~lh_ok)
    c[only_r] = xy[only_r, rh]

    # if neither hip is confident -> keep zeros (no centering effect)
    return c


def build_features(
    xy: np.ndarray,                 # (T,33,2)
    conf: np.ndarray,               # (T,33)
    fps: float,
    *,
    center: str = "pelvis",
    use_motion: bool = True,
    use_conf_channel: bool = True,
    motion_scale_by_fps: bool = True,
    conf_gate: float = 0.2,
    mask: Optional[np.ndarray] = None,  # (T,33) bool/0-1
) -> np.ndarray:
    """
    Build per-joint features (T,33,F) from xy/conf, matching training by default.

    Features:
      - rel_xy (always) : 2 dims
      - vel_xy (optional): 2 dims
      - conf (optional)  : 1 dim

    Gating:
      - If mask is provided, it is treated as 'valid joint' mask.
      - Otherwise, we derive a mask from conf >= conf_gate.
      - Invalid joints are zeroed (xy and conf).
    """
    assert xy.ndim == 3 and xy.shape[1] == 33 and xy.shape[2] == 2
    assert conf.ndim == 2 and conf.shape[1] == 33

    xy = xy.astype(np.float32, copy=True)
    conf = conf.astype(np.float32, copy=True)

    if mask is None:
        m = (conf >= float(conf_gate))
    else:
        m = mask.astype(bool)

    # zero invalid joints
    xy[~m] = 0.0
    conf[~m] = 0.0

    if center == "pelvis":
        c = pelvis_center(xy, conf, gate=float(conf_gate))  # (T,2)
        rel = xy - c[:, None, :]
    elif center == "none":
        rel = xy
    else:
        raise ValueError("center must be 'pelvis' or 'none'")

    parts = [rel]

    if use_motion:
        vel = np.zeros_like(rel, dtype=np.float32)
        vel[1:] = rel[1:] - rel[:-1]
        if motion_scale_by_fps:
            vel *= float(fps)
        parts.append(vel)

    if use_conf_channel:
        parts.append(conf[..., None])

    feats = np.concatenate(parts, axis=-1).astype(np.float32)  # (T,33,F)
    return feats


def _label_from_npz(z: np.lib.npyio.NpzFile) -> Optional[float]:
    for k in ["y", "label", "target"]:
        if k in z:
            y = z[k]
            # y may be scalar or array([..])
            try:
                y = float(np.array(y).reshape(-1)[0])
                return y
            except Exception:
                pass
    return None


def _fps_from_npz(z: np.lib.npyio.NpzFile, fps_default: float) -> float:
    for k in ["fps", "fps_src", "fps_used"]:
        if k in z:
            try:
                return float(np.array(z[k]).reshape(-1)[0])
            except Exception:
                pass
    return float(fps_default)


class WindowDataset(Dataset):
    """
    Loads .npz windows from a directory.

    Supports both:
      - legacy: keys include 'x' (T,C) and 'y'
      - current: keys include 'xy' (T,33,2), 'conf' (T,33), optional 'mask' and 'fps'
    """
    def __init__(
        self,
        root_dir: str,
        arch: str,
        *,
        fps_default: float,
        center: str,
        use_motion: bool,
        use_conf_channel: bool,
        motion_scale_by_fps: bool,
        conf_gate: float,
        use_precomputed_mask: bool,
    ):
        self.root_dir = root_dir
        self.arch = arch
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under: {root_dir}")

        self.fps_default = fps_default
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
        z = np.load(path, allow_pickle=True)

        y = _label_from_npz(z)
        if y is None:
            # allow unlabeled; treat as -1
            y = -1.0
        fps = _fps_from_npz(z, self.fps_default)

        # Legacy windows (already-built x)
        if "x" in z:
            x = np.array(z["x"], dtype=np.float32)  # (T,C) for TCN
            if self.arch == "gcn":
                # attempt to reshape (T,V,F) if stored flattened (rare)
                if x.ndim == 2 and x.shape[1] % 33 == 0:
                    F = x.shape[1] // 33
                    x = x.reshape(x.shape[0], 33, F)
            return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32), torch.tensor([fps], dtype=torch.float32)

        # Current windows (xy/conf)
        if "xy" not in z or "conf" not in z:
            raise KeyError(f"{path} is missing required keys. Found: {list(z.keys())}")

        xy = np.array(z["xy"], dtype=np.float32)    # (T,33,2)
        conf = np.array(z["conf"], dtype=np.float32)  # (T,33)

        mask = None
        if self.use_precomputed_mask and "mask" in z:
            mask = np.array(z["mask"])

        feats = build_features(
            xy, conf, fps,
            center=self.center,
            use_motion=self.use_motion,
            use_conf_channel=self.use_conf_channel,
            motion_scale_by_fps=self.motion_scale_by_fps,
            conf_gate=self.conf_gate,
            mask=mask,
        )  # (T,33,F)

        if self.arch == "tcn":
            T, V, F = feats.shape
            x = feats.reshape(T, V * F)
        else:
            x = feats  # (T,33,F)

        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32), torch.tensor([fps], dtype=torch.float32)


def collate(batch):
    xs, ys, fps = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B, ...)
    y = torch.cat(ys, dim=0)    # (B,)
    fps = torch.cat(fps, dim=0) # (B,)
    return x, y, fps


# -------------------------
# Inference + sweep + OPS
# -------------------------

@dataclass
class SweepResult:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    fa_per_24h: np.ndarray  # estimated on negative duration


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def predict_logits_dir(
    model: nn.Module,
    arch: str,
    val_dir: str,
    *,
    batch: int,
    device: torch.device,
    fps_default: float,
    center: str,
    use_motion: bool,
    use_conf_channel: bool,
    motion_scale_by_fps: bool,
    conf_gate: float,
    use_precomputed_mask: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = meta or {}

    ds = WindowDataset(
        val_dir, arch,
        fps_default=fps_default,
        center=center,
        use_motion=use_motion,
        use_conf_channel=use_conf_channel,
        motion_scale_by_fps=motion_scale_by_fps,
        conf_gate=conf_gate,
        use_precomputed_mask=use_precomputed_mask,
    )
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)

    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    fps_all: List[np.ndarray] = []

    for x, y, fps in dl:
        x = x.to(device)
        if arch == "gcn":
            # ensure shape (B,T,V,F)
            if x.ndim == 3:
                x = x.unsqueeze(-1)  # shouldn't happen, but keep safe
        if arch == "gcn" and meta.get("two_stream"):
            # Split [B,T,V,F] into joint-stream and motion-stream (+ optional conf).
            F_full = x.shape[-1]
            has_conf = (F_full == 5)
            xj = x[..., 0:2]
            xm = x[..., 2:4]
            if has_conf:
                conf = x[..., 4:5]
                xj = torch.cat([xj, conf], dim=-1)
                xm = torch.cat([xm, conf], dim=-1)
            out = model(xj, xm)
        else:
            out = model(x)
        # Normalise to shape (B,) for threshold fitting
        if isinstance(out, torch.Tensor):
            if out.ndim == 2 and out.shape[1] == 1:
                out = out[:, 0]
            elif out.ndim == 2 and out.shape[1] == 2:
                raise ValueError("fit_ops.py expects a single-logit binary model (B,1), got (B,2).")
        logits_all.append(out.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
        fps_all.append(fps.detach().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    logits = np.asarray(logits).reshape(-1)
    y = np.concatenate(y_all, axis=0)
    fps = np.concatenate(fps_all, axis=0)
    return logits, y, fps


def _fa_per_24h(fp: int, neg_duration_sec: float) -> float:
    if neg_duration_sec <= 0:
        return float("inf") if fp > 0 else 0.0
    return (fp / neg_duration_sec) * 86400.0


def sweep_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    *,
    stride_frames: int,
    fps: np.ndarray,
    n_grid: int = 400,
) -> SweepResult:
    """
    Sweep thresholds and compute curve metrics.

    FA/24h is estimated by assuming each window corresponds to 'stride' seconds of time:
        neg_duration_sec ~= sum_{negative windows} (stride_frames / fps_i)
    """
    assert probs.shape == y.shape
    pos = (y > 0.5)
    neg = (y <= 0.5)

    # Duration of negative data covered
    neg_duration_sec = float(np.sum(stride_frames / np.maximum(fps[neg], 1e-6)))

    # Threshold candidates (grid in [0,1])
    thresholds = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)

    precision = np.zeros_like(thresholds, dtype=np.float32)
    recall = np.zeros_like(thresholds, dtype=np.float32)
    f1 = np.zeros_like(thresholds, dtype=np.float32)
    fa24 = np.zeros_like(thresholds, dtype=np.float32)

    P = int(np.sum(pos))
    N = int(np.sum(neg))

    for i, thr in enumerate(thresholds):
        pred = probs >= thr
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & neg))
        fn = P - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / P if P > 0 else 0.0
        f1_i = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision[i] = prec
        recall[i] = rec
        f1[i] = f1_i
        fa24[i] = _fa_per_24h(fp, neg_duration_sec)

    return SweepResult(thresholds=thresholds, precision=precision, recall=recall, f1=f1, fa_per_24h=fa24)


def _pick_min_fa_under_recall(s: SweepResult, recall_floor: float) -> int:
    ok = np.where(s.recall >= recall_floor)[0]
    if ok.size == 0:
        # fallback: max recall
        return int(np.argmax(s.recall))
    # pick minimal FA; tie-breaker: higher recall then higher F1
    best = ok[np.argmin(s.fa_per_24h[ok])]
    # refine ties
    ties = ok[np.where(s.fa_per_24h[ok] == s.fa_per_24h[best])[0]]
    if ties.size > 1:
        best = ties[np.argmax(s.recall[ties])]
        ties2 = ties[np.where(s.recall[ties] == s.recall[best])[0]]
        if ties2.size > 1:
            best = ties2[np.argmax(s.f1[ties2])]
    return int(best)


def choose_ops(s: SweepResult, *, recall_floor_op1: float, recall_floor_op3: float) -> Dict[str, int]:
    # OP2: max F1 (tie: lower FA)
    op2 = int(np.argmax(s.f1))
    ties = np.where(s.f1 == s.f1[op2])[0]
    if ties.size > 1:
        op2 = int(ties[np.argmin(s.fa_per_24h[ties])])

    op1 = _pick_min_fa_under_recall(s, recall_floor_op1)
    op3 = _pick_min_fa_under_recall(s, recall_floor_op3)

    return {"OP1_high_recall": op1, "OP2_balanced": op2, "OP3_low_alarm": op3}


def _format_op(s: SweepResult, idx: int) -> dict:
    return {
        "thr": float(s.thresholds[idx]),
        "precision": float(s.precision[idx]),
        "recall": float(s.recall[idx]),
        "f1": float(s.f1[idx]),
        "fa_per_24h": float(s.fa_per_24h[idx]),
    }


# -------------------------
# CLI
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["tcn", "gcn"])
    ap.add_argument("--val_dir", required=True, help="Validation windows dir (e.g. .../val)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (best.pt)")
    ap.add_argument("--out", required=True, help="Output YAML path (ops file)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=33724876)

    # Loading behavior
    ap.add_argument("--strict", type=int, default=1, help="1=strict state_dict load (default), 0=allow missing keys")
    ap.add_argument("--fps_default", type=float, default=30.0, help="Used if window npz has no fps key")

    # Preprocessing knobs (must match training/checkpoint input dims)
    ap.add_argument("--center", default="pelvis", choices=["pelvis", "none"])
    ap.add_argument("--use_motion", default=None, help="1/0; if omitted, inferred from ckpt cfg when possible")
    ap.add_argument("--use_conf_channel", default=None, help="1/0; if omitted, inferred from ckpt cfg when possible")
    ap.add_argument("--motion_scale_by_fps", default=None, help="1/0; if omitted, inferred from ckpt cfg when possible")
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--use_precomputed_mask", default="0", help="1/0; if 1 and window has 'mask', use it")

    # Operating point preferences
    ap.add_argument("--recall_floor_op1", type=float, default=0.98)
    ap.add_argument("--recall_floor_op3", type=float, default=0.70)
    ap.add_argument("--grid", type=int, default=400, help="Number of thresholds in sweep grid")

    args = ap.parse_args()

    seed_all(args.seed)
    device = pick_device()

    strict = bool(int(args.strict))
    model, cfg, meta = build_model_from_ckpt(args.ckpt, args.arch, device, strict=strict)

    # Infer preprocessing defaults from ckpt cfg (if present)
    def _auto_bool(v, fallback: bool) -> bool:
        if v is None:
            return fallback
        return bool(int(v))

    # Our training cfg stores use_motion/use_conf_channel.
    if "use_motion" in cfg:
        use_motion_ckpt = bool(int(cfg.get("use_motion", 1)))
    elif "use_vel" in cfg:
        use_motion_ckpt = bool(int(cfg.get("use_vel", 1)))
    else:
        use_motion_ckpt = True

    use_conf_ckpt = bool(int(cfg.get("use_conf_channel", 1))) if "use_conf_channel" in cfg else True

    # motion_scale_by_fps wasn't in old cfg; default True (matches training).
    motion_scale_ckpt = bool(int(cfg.get("motion_scale_by_fps", 1))) if "motion_scale_by_fps" in cfg else True

    use_motion = _auto_bool(args.use_motion, use_motion_ckpt)
    use_conf_channel = _auto_bool(args.use_conf_channel, use_conf_ckpt)
    motion_scale_by_fps = _auto_bool(args.motion_scale_by_fps, motion_scale_ckpt)
    use_precomputed_mask = bool(int(args.use_precomputed_mask))

    # Dimension sanity check (prevents silent mismatches)
    if args.arch == "tcn":
        # Expected in_ch = 33 * (2 + 2*use_motion + 1*use_conf_channel)
        F = 2 + (2 if use_motion else 0) + (1 if use_conf_channel else 0)
        expected_in_ch = 33 * F
        ckpt_in_ch = int(meta.get("in_ch", expected_in_ch))
        if expected_in_ch != ckpt_in_ch:
            raise SystemExit(
                f"[err] TCN feature dim mismatch: build_features gives in_ch={expected_in_ch}, "
                f"but checkpoint expects in_ch={ckpt_in_ch}. "
                f"Fix flags: --use_motion / --use_conf_channel."
            )
    else:
        F = 2 + (2 if use_motion else 0) + (1 if use_conf_channel else 0)
        ckpt_in_feats = int(meta.get("in_feats", F))
        if F != ckpt_in_feats:
            raise SystemExit(
                f"[err] GCN feature dim mismatch: build_features gives in_feats={F}, "
                f"but checkpoint expects in_feats={ckpt_in_feats}. "
                f"Fix flags: --use_motion / --use_conf_channel."
            )

    stride_frames = _parse_stride_from_dir(args.val_dir)
    if stride_frames is None:
        raise SystemExit(
            f"[err] Cannot parse stride from val_dir path: {args.val_dir}\n"
            f"      Expected something like .../windows_W48_S12/val"
        )

    logits, y, fps = predict_logits_dir(
        model, args.arch, args.val_dir,
        batch=args.batch,
        device=device,
        fps_default=args.fps_default,
        center=args.center,
        use_motion=use_motion,
        use_conf_channel=use_conf_channel,
        motion_scale_by_fps=motion_scale_by_fps,
        conf_gate=args.conf_gate,
        use_precomputed_mask=use_precomputed_mask,
        meta=meta,
    )

    # Filter unlabeled
    keep = (y >= 0.0)
    logits = logits[keep]
    y = y[keep]
    fps = fps[keep]

    probs = sigmoid(logits.astype(np.float64))
    sweep = sweep_thresholds(
        probs.astype(np.float32),
        y.astype(np.float32),
        stride_frames=stride_frames,
        fps=fps.astype(np.float32),
        n_grid=int(args.grid),
    )

    picks = choose_ops(sweep, recall_floor_op1=args.recall_floor_op1, recall_floor_op3=args.recall_floor_op3)

    out = {
        "arch": args.arch,
        "val_dir": args.val_dir,
        "ckpt": args.ckpt,
        "stride_frames": int(stride_frames),
        "preproc": {
            "center": args.center,
            "use_motion": int(use_motion),
            "use_conf_channel": int(use_conf_channel),
            "motion_scale_by_fps": int(motion_scale_by_fps),
            "conf_gate": float(args.conf_gate),
            "use_precomputed_mask": int(use_precomputed_mask),
            "fps_default": float(args.fps_default),
        },
        "ops": {
            "OP1_high_recall": _format_op(sweep, picks["OP1_high_recall"]),
            "OP2_balanced": _format_op(sweep, picks["OP2_balanced"]),
            "OP3_low_alarm": _format_op(sweep, picks["OP3_low_alarm"]),
        },
    }

    yaml_dump_simple(out, args.out)
    print(f"[ok] wrote ops -> {args.out}")
    print("[ok] OP1:", out["ops"]["OP1_high_recall"])
    print("[ok] OP2:", out["ops"]["OP2_balanced"])
    print("[ok] OP3:", out["ops"]["OP3_low_alarm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())