#!/usr/bin/env python3
# server/app.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import get_conn


# ===================================================
# Window config
# ===================================================
WIN_W = 48
WIN_S = 12


# ===================================================
# Paths (window dirs)
# ===================================================
WIN_DIR_LE2I = Path(f"data/processed/le2i/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_URFD = Path(f"data/processed/urfd/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_CAUCA = Path(f"data/processed/caucafall/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_MUVIM = Path(f"data/processed/muvim/windows_W{WIN_W}_S{WIN_S}")  # if you have it


# ===================================================
# Checkpoints (update/add as needed)
# ===================================================
CKPT_LE2I_TCN = Path(f"outputs/le2i_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_URFD_TCN = Path(f"outputs/urfd_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_CAUCA_TCN = Path(f"outputs/caucafall_tcn_W{WIN_W}S{WIN_S}/best.pt")

# Example GCN you used earlier:
CKPT_LE2I_GCN = Path(f"outputs/le2i_gcn_W{WIN_W}S{WIN_S}/best.pt")

# Add if/when you train these:
CKPT_URFD_GCN = Path(f"outputs/urfd_gcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_CAUCA_GCN = Path(f"outputs/caucafall_gcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_MUVIM_TCN = Path(f"outputs/muvim_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_MUVIM_GCN = Path(f"outputs/muvim_gcn_W{WIN_W}S{WIN_S}/best.pt")


# ===================================================
# Ops YAMLs (update names to match your latest fit_ops output)
# ===================================================
OPS_LE2I_TCN = Path("configs/ops_le2i_tcn.yaml")
OPS_URFD_TCN = Path("configs/ops_urfd_tcn.yaml")
OPS_CAUCA_TCN = Path("configs/ops_caucafall_tcn.yaml")

OPS_LE2I_GCN = Path("configs/ops_le2i_gcn.yaml")
OPS_URFD_GCN = Path("configs/ops_urfd_gcn.yaml")
OPS_CAUCA_GCN = Path("configs/ops_caucafall_gcn.yaml")
OPS_MUVIM_TCN = Path("configs/ops_muvim_tcn.yaml")
OPS_MUVIM_GCN = Path("configs/ops_muvim_gcn.yaml")


# ===================================================
# Metrics reports (unchanged)
# ===================================================
REPORT_LE2I = Path("outputs/reports/le2i_in_domain.json")
REPORT_URFD_CROSS = Path("outputs/reports/urfd_cross.json")
REPORT_CAUCA_IN_DOMAIN = Path("outputs/reports/caucafall_in_domain.json")
REPORT_CAUCA_ON_URFD = Path("outputs/reports/caucafall_on_urfd.json")

DEFAULT_RESIDENT_ID = 1


# ===================================================
# FastAPI app + CORS
# ===================================================
app = FastAPI(title="Elder Fall Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================
# Device
# ===================================================
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


# ===================================================
# Checkpoint helpers (shared)
# ===================================================
def _get_state_dict(ckpt: dict) -> dict:
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    raise KeyError("Checkpoint has no 'model' or 'state_dict' dict.")


def _infer_out_dim_from_head(sd: dict) -> int:
    w = sd.get("head.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return 1


def _infer_prob_mode(sd: dict) -> str:
    """
    If head out_dim == 1 => sigmoid
    If head out_dim == 2 => softmax
    """
    out_dim = _infer_out_dim_from_head(sd)
    return "sigmoid" if out_dim == 1 else "softmax"


# ===================================================
# TCN models (same structure as eval scripts)
# ===================================================
class SimpleTCN(nn.Module):
    def __init__(self, in_dim=66, hidden=128, dropout=0.2, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        # x: [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)
        return self.head(h)


class ResTCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, p: float = 0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        y = self.drop(self.relu(self.bn(self.conv(x))))
        return x + y


class EnhancedTCN(nn.Module):
    """
    Matches checkpoints with keys:
      conv_in.0.weight, blocks.N.conv.weight, head.weight, ...
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        kernel_in: int = 5,
        kernel_block: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        pad_in = (kernel_in - 1) // 2
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=kernel_in, padding=pad_in),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        if dilations is None:
            dilations = [1, 2, 4]
        self.blocks = nn.ModuleList(
            [ResTCNBlock(hidden, kernel_size=kernel_block, dilation=int(d), p=dropout) for d in dilations]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        # x: [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        h = x.mean(dim=-1)
        return self.head(h)


def _infer_num_blocks(sd: dict) -> int:
    idxs = []
    for k in sd.keys():
        if k.startswith("blocks.") and ".conv.weight" in k:
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0


def _infer_dilations(ckpt: dict, n_blocks: int) -> List[int]:
    d = ckpt.get("dilations", None)
    if isinstance(d, (list, tuple)) and len(d) == n_blocks:
        return [int(x) for x in d]
    return [2**i for i in range(n_blocks)] if n_blocks > 0 else [1, 2, 4]


def build_tcn_from_ckpt(ckpt: dict) -> Tuple[nn.Module, str]:
    sd = _get_state_dict(ckpt)

    # Enhanced TCN?
    if any(k.startswith("conv_in.") for k in sd.keys()) and any(k.startswith("blocks.") for k in sd.keys()):
        conv_w = sd["conv_in.0.weight"]  # [hidden, in_dim, k]
        hidden = int(conv_w.shape[0])
        in_dim = int(conv_w.shape[1])
        k_in = int(conv_w.shape[2])

        out_dim = int(sd["head.weight"].shape[0])  # 1 or 2
        n_blocks = _infer_num_blocks(sd)
        dilations = _infer_dilations(ckpt, n_blocks)

        model = EnhancedTCN(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            kernel_in=k_in,
            kernel_block=3,
            dilations=dilations,
            dropout=float(ckpt.get("dropout", 0.3)),
        )
        prob_mode = "sigmoid" if out_dim == 1 else "softmax"
        return model, prob_mode

    # Simple TCN fallback
    out_dim = _infer_out_dim_from_head(sd)
    in_dim = int(ckpt.get("in_ch", ckpt.get("in_dim", 66)))
    hidden = int(ckpt.get("hidden", 128))
    model = SimpleTCN(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    return model, ("sigmoid" if out_dim == 1 else "softmax")


# ===================================================
# GCN models (support both new + legacy checkpoints)
# ===================================================
def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),
        (11, 13), (13, 15), (15, 17),
        (12, 14), (14, 16), (16, 18),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (7, 28), (8, 27),
    ]
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_joints and 0 <= j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor):
        # x: [B,T,V,C]
        x = torch.einsum("ij,btjc->btic", A_hat, x)
        return self.lin(x)


class GCNBlockNew(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.2):
        super().__init__()
        self.gc = GraphConv(in_feats, out_feats)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor):
        return self.do(self.act(self.gc(x, A_hat)))


class GCNTemporalGRU(nn.Module):
    """
    New-style checkpoint keys:
      g1.gc.lin.*, g2.gc.lin.*, temporal.weight_ih_l0, ..., head.*
    """
    def __init__(self, num_joints: int, in_feats: int, hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        A = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A))
        self.g1 = GCNBlockNew(in_feats, hidden, dropout)
        self.g2 = GCNBlockNew(hidden, hidden, dropout)
        self.temporal = nn.GRU(input_size=num_joints * hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.g1(x, self.A_hat)
        x = self.g2(x, self.A_hat)
        b, t, v, c = x.shape
        x = x.reshape(b, t, v * c)
        _, h = self.temporal(x)
        return self.head(h.squeeze(0))


class GCNBlockLegacy(nn.Module):
    """
    Legacy keys:
      block1.lin.*, block1.bn.*, block2.lin.*, block2.bn.*
    """
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.2, prefix: str = "block"):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor):
        # x: [B,T,V,C]
        x = torch.einsum("ij,btjc->btic", A_hat, x)  # message pass
        x = self.lin(x)  # [B,T,V,H]

        b, t, v, h = x.shape
        x2 = x.reshape(b * t * v, h)
        x2 = self.bn(x2)
        x2 = self.act(x2)
        x2 = self.do(x2)
        return x2.reshape(b, t, v, h)


class GCNTemporalLegacy(nn.Module):
    """
    Legacy checkpoint keys seen in your error:
      block1.*, block2.*, temporal.0.weight/bias, head.weight/bias
    """
    def __init__(self, num_joints: int, in_feats: int, hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        A = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A))
        self.block1 = GCNBlockLegacy(in_feats, hidden, dropout)
        self.block2 = GCNBlockLegacy(hidden, hidden, dropout)

        # temporal is a Linear wrapped in Sequential so keys become temporal.0.weight/bias
        self.temporal = nn.Sequential(
            nn.Linear(num_joints * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        # x: [B,T,V,C]
        x = self.block1(x, self.A_hat)
        x = self.block2(x, self.A_hat)

        # simple temporal pooling then MLP
        b, t, v, h = x.shape
        x = x.reshape(b, t, v * h)      # [B,T,VH]
        x = x.mean(dim=1)               # [B,VH]
        x = self.temporal(x)            # [B,H]
        return self.head(x)             # [B,out_dim]


def build_gcn_from_ckpt(ckpt: dict) -> Tuple[nn.Module, str]:
    sd = _get_state_dict(ckpt)

    # Infer basics from state dict
    out_dim = _infer_out_dim_from_head(sd)
    prob_mode = "sigmoid" if out_dim == 1 else "softmax"

    # Prefer explicit metadata, fallback to inference
    num_joints = int(ckpt.get("num_joints", 33))

    # New GRU-style?
    if any(k.startswith("g1.") for k in sd.keys()) and any("temporal.weight_ih_l0" in k for k in sd.keys()):
        # infer in_feats from g1.gc.lin.weight
        w = sd.get("g1.gc.lin.weight")
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            raise RuntimeError("Cannot infer gcn in_feats/hidden from g1.gc.lin.weight")
        hidden = int(w.shape[0])
        in_feats = int(w.shape[1])

        model = GCNTemporalGRU(
            num_joints=num_joints,
            in_feats=in_feats,
            hidden=hidden,
            out_dim=out_dim,
            dropout=float(ckpt.get("dropout", 0.2)),
        )
        return model, prob_mode

    # Legacy block1/block2 + temporal.0 linear
    if any(k.startswith("block1.") for k in sd.keys()) and any(k.startswith("temporal.0.") for k in sd.keys()):
        w1 = sd.get("block1.lin.weight")
        if not isinstance(w1, torch.Tensor) or w1.ndim != 2:
            raise RuntimeError("Cannot infer gcn legacy in_feats/hidden from block1.lin.weight")
        hidden = int(w1.shape[0])
        in_feats = int(w1.shape[1])

        model = GCNTemporalLegacy(
            num_joints=num_joints,
            in_feats=in_feats,
            hidden=hidden,
            out_dim=out_dim,
            dropout=float(ckpt.get("dropout", 0.2)),
        )
        return model, prob_mode

    raise RuntimeError("Unknown GCN checkpoint structure (neither new nor legacy).")


# ===================================================
# Unified model loader (TCN or GCN)
# ===================================================
def load_model_from_ckpt(ckpt_path: Path, arch: str) -> Tuple[nn.Module, str, float]:
    """
    Returns (model, prob_mode, default_thr_from_ckpt).
    """
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)} at {ckpt_path}")

    sd = _get_state_dict(ckpt)
    default_thr = float(ckpt.get("best_thr", 0.5))

    if arch == "tcn":
        model, prob_mode = build_tcn_from_ckpt(ckpt)
    elif arch == "gcn":
        model, prob_mode = build_gcn_from_ckpt(ckpt)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model, prob_mode, default_thr


def load_ops_thr(path: Path, default_thr: float) -> float:
    """
    Loads threshold from ops yaml.
    Prefers OP3_low_alarm.thr, otherwise default_thr.
    """
    if not path.exists():
        return default_thr
    with open(path, "r", encoding="utf-8") as f:
        ops = yaml.safe_load(f)
    if isinstance(ops, dict):
        op3 = ops.get("OP3_low_alarm")
        if isinstance(op3, dict) and "thr" in op3:
            return float(op3["thr"])
    return default_thr


# ===================================================
# Inference feature builders
# ===================================================
def build_tcn_features_from_xyconf(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    # xy [T,33,2], conf [T,33] -> x [T,66]
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = xy * conf[..., None]
    return x.reshape(x.shape[0], -1).astype(np.float32)


def build_gcn_features_from_xyconf(xy: np.ndarray, conf: np.ndarray, fps: float) -> np.ndarray:
    """
    xy [T,33,2], conf [T,33] -> feats [T,33,4]
    feats = [pelvis-centered xy, velocity]
    """
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = xy * conf[..., None]  # [T,33,2]

    pelvis = x[:, 23:24, :]   # [T,1,2]
    x_rel = x - pelvis        # [T,33,2]

    vel = np.zeros_like(x_rel)
    vel[1:] = (x_rel[1:] - x_rel[:-1]) * float(fps)

    feats = np.concatenate([x_rel, vel], axis=-1)  # [T,33,4]
    return feats.astype(np.float32)


@torch.no_grad()
def p_fall_from_logits(logits: torch.Tensor, prob_mode: str) -> np.ndarray:
    """
    logits: [B,1] or [B,2] or [B]
    returns p_fall: [B]
    """
    if prob_mode == "sigmoid":
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return torch.sigmoid(logits).detach().cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


# ===================================================
# Model registry (supports multiple arch per dataset)
# ===================================================
class ModelSpec(BaseModel):
    id: str                 # e.g. "le2i_tcn", "le2i_gcn"
    label: str
    dataset: str
    arch: str               # "tcn" | "gcn"
    ckpt: str
    ops: Optional[str] = None
    window_dir: Optional[str] = None
    fps_default: float = 30.0  # used for GCN velocity if needed (and for live stream)


# Define what you want to expose to the UI
MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        id="le2i_tcn",
        label="LE2I (TCN)",
        dataset="LE2I",
        arch="tcn",
        ckpt=str(CKPT_LE2I_TCN),
        ops=str(OPS_LE2I_TCN),
        window_dir=str(WIN_DIR_LE2I),
        fps_default=25.0,
    ),
    ModelSpec(
        id="le2i_gcn",
        label="LE2I (GCN)",
        dataset="LE2I",
        arch="gcn",
        ckpt=str(CKPT_LE2I_GCN),
        ops=str(OPS_LE2I_GCN),
        window_dir=str(WIN_DIR_LE2I),
        fps_default=25.0,
    ),
    ModelSpec(
        id="urfd_tcn",
        label="URFD (TCN)",
        dataset="URFD",
        arch="tcn",
        ckpt=str(CKPT_URFD_TCN),
        ops=str(OPS_URFD_TCN),
        window_dir=str(WIN_DIR_URFD),
        fps_default=30.0,
    ),
    ModelSpec(
        id="caucafall_tcn",
        label="CAUCAFall (TCN)",
        dataset="CAUCAFall",
        arch="tcn",
        ckpt=str(CKPT_CAUCA_TCN),
        ops=str(OPS_CAUCA_TCN),
        window_dir=str(WIN_DIR_CAUCA),
        fps_default=23.0,
    ),
    # Optional MUViM (won't load unless ckpt exists)
    ModelSpec(
        id="muvim_tcn",
        label="MUVIM (TCN)",
        dataset="MUVIM",
        arch="tcn",
        ckpt=str(CKPT_MUVIM_TCN),
        ops=str(OPS_MUVIM_TCN),
        window_dir=str(WIN_DIR_MUVIM),
        fps_default=30.0,
    ),
]


# Runtime registry (filled at startup)
MODELS: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def _startup_load_models():
    """
    Load available models. If a ckpt is missing, skip it (so server still starts).
    """
    global MODELS
    MODELS = {}

    for spec in MODEL_SPECS:
        ckpt_path = Path(spec.ckpt)
        if not ckpt_path.exists():
            # Skip silently (or print if you prefer)
            print(f"[startup] skip missing ckpt: {ckpt_path}")
            continue

        try:
            model, prob_mode, thr_ckpt = load_model_from_ckpt(ckpt_path, arch=spec.arch)
            thr_ui = load_ops_thr(Path(spec.ops) if spec.ops else Path(""), thr_ckpt)

            MODELS[spec.id] = {
                "spec": spec.model_dump(),
                "model": model,
                "prob_mode": prob_mode,
                "thr_ckpt": thr_ckpt,
                "thr_ui": thr_ui,
            }
            print(f"[startup] loaded {spec.id} arch={spec.arch} prob={prob_mode} thr_ui={thr_ui:.3f}")
        except Exception as e:
            print(f"[startup] FAILED loading {spec.id} ({spec.ckpt}): {e}")


# ===================================================
# Metrics report helpers (unchanged)
# ===================================================
def load_report(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarise_report(rep):
    if rep is None or not isinstance(rep, dict):
        return []
    ops_dict = rep["ops"] if "ops" in rep and isinstance(rep["ops"], dict) else rep

    summary = []
    for name, d in ops_dict.items():
        if not isinstance(d, dict):
            continue
        summary.append(
            {
                "name": name,
                "thr": d.get("thr"),
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "fa24h": d.get("fa24h"),
                "F1": d.get("F1"),
            }
        )
    return summary


# ===================================================
# Demo window loading (supports both TCN + GCN)
# ===================================================
def make_window_features_from_npz(npz_path: Path, arch: str, fps_default: float) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=False)
    xy = d["xy"].astype(np.float32)      # [T,33,2]
    conf = d["conf"].astype(np.float32)  # [T,33]
    fps = float(d["fps"]) if "fps" in d.files else float(fps_default)

    if arch == "tcn":
        return build_tcn_features_from_xyconf(xy, conf)  # [T,66]
    return build_gcn_features_from_xyconf(xy, conf, fps=fps)  # [T,33,4]


def load_sample_windows(root_dir: Path, arch: str, fps_default: float, max_windows: int = 40) -> np.ndarray:
    root = root_dir / "test"
    npz_files = sorted(root.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz windows found in {root}")

    xs = []
    for f in npz_files[:max_windows]:
        xs.append(make_window_features_from_npz(f, arch=arch, fps_default=fps_default))

    X = np.stack(xs, axis=0)  # [N,T,C] or [N,T,V,C]
    return X


def run_sequence(model: nn.Module, prob_mode: str, X: np.ndarray, thr: float):
    with torch.no_grad():
        inp = torch.from_numpy(X).float().to(DEVICE)
        logits = model(inp)
        probs = p_fall_from_logits(logits, prob_mode=prob_mode)

    points = []
    for i, p in enumerate(probs):
        points.append({"t": int(i), "p_fall": float(p), "fall": bool(p >= thr)})
    return points


# ===================================================
# API endpoints
# ===================================================
@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/api/models/summary")
def models_summary():
    """
    Summary of all loaded models (TCN + GCN).
    """
    le2i_rep = summarise_report(load_report(REPORT_LE2I))
    urfd_cross_rep = summarise_report(load_report(REPORT_URFD_CROSS))
    cauca_in_rep = summarise_report(load_report(REPORT_CAUCA_IN_DOMAIN))
    cauca_on_urfd_rep = summarise_report(load_report(REPORT_CAUCA_ON_URFD))

    models = []
    for mid, entry in MODELS.items():
        spec = entry["spec"]
        models.append(
            {
                "id": mid,
                "label": spec["label"],
                "dataset": spec["dataset"],
                "arch": spec["arch"],
                "ckpt": spec["ckpt"],
                "prob_mode": entry["prob_mode"],
                "best_thr_from_ckpt": entry["thr_ckpt"],
                "ui_threshold": entry["thr_ui"],
            }
        )

    return {
        "models": models,
        "reports": {
            "le2i_in_domain": le2i_rep,
            "le2i_cross_to_urfd": urfd_cross_rep,
            "caucafall_in_domain": cauca_in_rep,
            "caucafall_on_urfd": cauca_on_urfd_rep,
        },
    }


@app.post("/api/demo/{model_id}_fall")
def monitor_sequence(model_id: str):
    """
    Replay a short sequence through the selected model.
    model_id must exist in MODELS, e.g. le2i_tcn / le2i_gcn / urfd_tcn / ...
    """
    model_id = model_id.lower()
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    entry = MODELS[model_id]
    spec = entry["spec"]

    window_dir = spec.get("window_dir")
    if not window_dir:
        raise HTTPException(status_code=400, detail="This model has no window_dir configured")

    X = load_sample_windows(
        Path(window_dir),
        arch=spec["arch"],
        fps_default=float(spec.get("fps_default", 30.0)),
        max_windows=40,
    )

    points = run_sequence(
        entry["model"],
        entry["prob_mode"],
        X,
        float(entry["thr_ui"]),
    )

    return {
        "fps": 5,
        "threshold": float(entry["thr_ui"]),
        "points": points,
        "arch": spec["arch"],
        "model_id": model_id,
    }


# ----------------------------
# Realtime inference payload
# ----------------------------
class PoseWindowPayload(BaseModel):
    model_id: str = "le2i_tcn"  # now flexible
    xy: List[List[List[float]]]
    conf: List[List[float]]


@app.post("/api/monitor/predict_window")
def predict_window(payload: PoseWindowPayload):
    model_id = payload.model_id.lower()

    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    entry = MODELS[model_id]
    spec = entry["spec"]
    arch = spec["arch"]
    thr = float(entry["thr_ui"])
    prob_mode = entry["prob_mode"]
    model = entry["model"]
    fps_default = float(spec.get("fps_default", 30.0))

    xy = np.array(payload.xy, dtype=np.float32)      # [T,33,2]
    conf = np.array(payload.conf, dtype=np.float32)  # [T,33]

    if xy.ndim != 3 or xy.shape[1] != 33 or xy.shape[2] != 2:
        raise HTTPException(status_code=400, detail=f"xy must have shape [T,33,2], got {xy.shape}")
    if conf.shape != (xy.shape[0], 33):
        raise HTTPException(status_code=400, detail=f"conf must have shape [T,33], got {conf.shape}")

    # Build features exactly like training/eval
    if arch == "tcn":
        x = build_tcn_features_from_xyconf(xy, conf)          # [T,66]
        X = x[None, ...]                                      # [1,T,66]
    else:
        feats = build_gcn_features_from_xyconf(xy, conf, fps=fps_default)  # [T,33,4]
        X = feats[None, ...]                                   # [1,T,33,4]

    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(DEVICE))
        p = p_fall_from_logits(logits, prob_mode=prob_mode)[0]

    return {
        "model_id": model_id,
        "arch": arch,
        "threshold": thr,
        "p_fall": float(p),
        "fall": bool(p >= thr),
        "prob_mode": prob_mode,
        "device": str(DEVICE),
    }


# ===================================================
# DB-backed endpoints (unchanged from your version)
# ===================================================
@app.get("/api/dashboard/summary")
def dashboard_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, display_name FROM residents WHERE id=%s", (resident_id,))
            resident = cur.fetchone()
            if resident is None:
                raise HTTPException(status_code=404, detail="Resident not found")

            cur.execute(
                """
                SELECT monitoring_enabled, api_online, last_latency_ms, active_model_id
                FROM system_settings
                WHERE resident_id = %s
                """,
                (resident_id,),
            )
            settings = cur.fetchone() or {}

            active_model_name = None
            if settings.get("active_model_id"):
                cur.execute("SELECT name FROM models WHERE id=%s", (settings["active_model_id"],))
                m = cur.fetchone()
                if m:
                    active_model_name = m["name"]

            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN type='fall'
                         AND status='confirmed_fall' THEN 1 ELSE 0 END) AS falls,
                  SUM(CASE WHEN status='false_alarm' THEN 1 ELSE 0 END) AS false_alarms,
                  SUM(CASE WHEN status IN ('pending_review','confirmed_fall')
                           THEN 1 ELSE 0 END) AS alerts
                FROM events
                WHERE resident_id = %s
                  AND DATE(event_time) = CURDATE()
                """,
                (resident_id,),
            )
            stats = cur.fetchone() or {}
            falls_today = stats.get("falls") or 0
            false_alarms_today = stats.get("false_alarms") or 0
            alerts_today = stats.get("alerts") or 0

    status = "alert" if alerts_today > 0 else "normal"

    return {
        "resident": {"id": resident["id"], "name": resident["display_name"]},
        "status": status,
        "today": {
            "falls_detected": int(falls_today),
            "false_alarms": int(false_alarms_today),
        },
        "system": {
            "model_name": active_model_name,
            "monitoring_enabled": bool(settings.get("monitoring_enabled", 1)),
            "api_online": bool(settings.get("api_online", 1)),
            "last_latency_ms": settings.get("last_latency_ms"),
        },
    }


@app.get("/api/events")
def list_events(
    resident_id: int = DEFAULT_RESIDENT_ID,
    limit: int = Query(50, ge=1, le=500),
):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  e.id,
                  e.event_time,
                  e.type,
                  e.p_fall,
                  e.delay_seconds,
                  e.status,
                  m.name AS model_name,
                  m.code AS model_code
                FROM events e
                JOIN models m ON e.model_id = m.id
                WHERE e.resident_id = %s
                ORDER BY e.event_time DESC
                LIMIT %s
                """,
                (resident_id, limit),
            )
            rows = cur.fetchall() or []

    events = []
    for r in rows:
        events.append(
            {
                "id": r["id"],
                "time": r["event_time"].isoformat() if isinstance(r["event_time"], datetime) else r["event_time"],
                "type": r["type"],
                "model_name": r["model_name"],
                "model_code": r["model_code"],
                "p_fall": float(r["p_fall"]) if r["p_fall"] is not None else None,
                "delay_seconds": float(r["delay_seconds"]) if r["delay_seconds"] is not None else None,
                "status": r["status"],
            }
        )

    return {"events": events}


@app.get("/api/events/summary")
def events_summary(resident_id: int = DEFAULT_RESIDENT_ID):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN type='fall'
                           AND status='confirmed_fall' THEN 1 ELSE 0 END) AS falls_today,
                  SUM(CASE WHEN status='false_alarm'
                           THEN 1 ELSE 0 END) AS false_alarms_today,
                  AVG(delay_seconds) AS avg_delay
                FROM events
                WHERE resident_id = %s
                  AND DATE(event_time) = CURDATE()
                """,
                (resident_id,),
            )
            row = cur.fetchone() or {}

    falls_today = row.get("falls_today") or 0
    false_alarms_today = row.get("false_alarms_today") or 0
    avg_delay = row.get("avg_delay")

    fa24h_estimate = float(false_alarms_today)

    return {
        "falls_today": int(falls_today),
        "false_alarms_today": int(false_alarms_today),
        "fa24h_estimate": fa24h_estimate,
        "avg_detection_delay": float(avg_delay) if avg_delay is not None else None,
    }


@app.get("/api/settings")
def get_settings(resident_id: int = DEFAULT_RESIDENT_ID):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  r.id AS resident_id,
                  r.display_name,
                  s.monitoring_enabled,
                  s.notify_on_every_fall,
                  s.require_confirmation,
                  s.store_event_clips,
                  s.anonymize_skeleton_data,
                  s.active_model_id,
                  s.active_operating_point,
                  m.name AS model_name,
                  op.name AS op_name
                FROM residents r
                LEFT JOIN system_settings s ON s.resident_id = r.id
                LEFT JOIN models m ON s.active_model_id = m.id
                LEFT JOIN operating_points op ON s.active_operating_point = op.id
                WHERE r.id = %s
                """,
                (resident_id,),
            )
            main = cur.fetchone()
            if not main:
                raise HTTPException(status_code=404, detail="Resident not found")

            cur.execute(
                """
                SELECT id, name, email, phone
                FROM caregivers
                WHERE resident_id = %s AND is_primary = 1
                ORDER BY id ASC
                LIMIT 1
                """,
                (resident_id,),
            )
            cg = cur.fetchone()

    return {
        "resident": {"id": main["resident_id"], "name": main["display_name"]},
        "caregiver": {
            "id": cg["id"] if cg else None,
            "name": cg["name"] if cg else None,
            "email": cg["email"] if cg else None,
            "phone": cg["phone"] if cg else None,
        },
        "notifications": {
            "notify_on_every_fall": bool(main.get("notify_on_every_fall", 1)),
            "require_confirmation": bool(main.get("require_confirmation", 0)),
        },
        "model": {
            "active_model_id": main.get("active_model_id"),
            "active_model_name": main.get("model_name"),
            "active_operating_point_id": main.get("active_operating_point"),
            "active_operating_point_name": main.get("op_name"),
        },
        "privacy": {
            "store_event_clips": bool(main.get("store_event_clips", 0)),
            "anonymize_skeleton_data": bool(main.get("anonymize_skeleton_data", 1)),
        },
        "monitoring_enabled": bool(main.get("monitoring_enabled", 1)),
    }
