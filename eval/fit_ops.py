
"""
Fit operating points (OP1/OP2/OP3) on a labelled validation set.

Supports:
  - TCN windows : x [T, 66]    (flattened 33x2, gated by conf)
  - GCN windows : x [T, 33, 4] (pelvis-centred xy + velocity, gated by conf)

OP definitions (same for both architectures)
--------------------------------------------
- OP1 (high recall):   minimise threshold subject to recall >= r1 (default 0.95)
- OP2 (balanced):      maximise F1
- OP3 (low alarm):     minimise FPR subject to recall >= r3 (default 0.90)

Notes
-----
- TCN checkpoints may be either:
    (A) "simple TCN" with keys like net.* and a 2-logit softmax head, OR
    (B) "enhanced TCN" with keys like conv_in.*, blocks.*, head.* and a 1-logit
        head trained with BCEWithLogitsLoss. In this case p(fall)=sigmoid(logit).
- GCN checkpoints may be either:
    (A) Conv-temporal GCN (from models/train_gcn.py in your repo):
        keys like block1.*, block2.*, temporal.0.*, head.* (often 1-logit)
    (B) GRU-temporal GCN (older/alternate):
        keys like g1.*, g2.*, temporal.weight_ih_l0, head.* (often 2-logit)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# -------------------------------------------------------------
# Datasets
# -------------------------------------------------------------
def _label_from_npz(d: np.lib.npyio.NpzFile) -> int:
    """
    Prefer numeric y, else label. Returns -1 for unlabeled.
    """
    if "y" in d.files:
        y = d["y"]
        if isinstance(y, np.ndarray):
            y = y.item() if y.shape == () else y.ravel()[0]
        try:
            y = float(y)
        except Exception:
            return -1
        if y < 0:
            return -1
        return 1 if y >= 0.5 else 0

    if "label" in d.files:
        lab = d["label"]
        if isinstance(lab, bytes):
            lab = lab.decode()
        if isinstance(lab, np.ndarray):
            lab = lab.item() if lab.shape == () else lab.ravel()[0]
        # numeric?
        try:
            v = float(lab)
            if v < 0:
                return -1
            return 1 if v >= 0.5 else 0
        except Exception:
            pass
        s = str(lab).lower()
        if s in ("fall", "1", "true", "pos", "positive"):
            return 1
        if s in ("adl", "0", "false", "neg", "negative", "normal", "nofall", "no_fall", "nonfall"):
            return 0
        return -1

    return -1


class WindowNPZ(Dataset):
    """TCN windows: returns (x[T,66], y)."""
    def __init__(self, root: str, skip_unlabeled: bool = True):
        self.files = sorted([str(p) for p in Path(root).glob("*.npz")])
        kept: List[str] = []
        for p in self.files:
            with np.load(p, allow_pickle=False) as d:
                y = _label_from_npz(d)
            if (not skip_unlabeled) or (y in (0, 1)):
                kept.append(p)
        self.files = kept

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        with np.load(p, allow_pickle=False) as d:
            xy = d["xy"].astype(np.float32)      # [T,33,2]
            conf = d["conf"].astype(np.float32)  # [T,33]
            y = _label_from_npz(d)

        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        x = (xy * conf[..., None]).reshape(xy.shape[0], -1)  # [T,66]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class WindowNPZGraph(Dataset):
    """GCN windows: returns (x[T,33,4], y)."""
    def __init__(self, root: str, skip_unlabeled: bool = True, fps_default: float = 30.0):
        self.files = sorted([str(p) for p in Path(root).glob("*.npz")])
        kept: List[str] = []
        for p in self.files:
            with np.load(p, allow_pickle=False) as d:
                y = _label_from_npz(d)
            if (not skip_unlabeled) or (y in (0, 1)):
                kept.append(p)
        self.files = kept
        self.fps_default = float(fps_default)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        with np.load(p, allow_pickle=False) as d:
            xy = d["xy"].astype(np.float32)      # [T,33,2]
            conf = d["conf"].astype(np.float32)  # [T,33]
            fps = float(d["fps"]) if ("fps" in d.files) else self.fps_default
            y = _label_from_npz(d)

        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        x = xy * conf[..., None]                # [T,33,2]
        pelvis = x[:, 23:24, :]                 # [T,1,2]
        x_rel = x - pelvis                      # [T,33,2]

        vel = np.zeros_like(x_rel)
        vel[1:] = (x_rel[1:] - x_rel[:-1]) * fps

        feats = np.concatenate([x_rel, vel], axis=-1)  # [T,33,4]
        return torch.from_numpy(feats), torch.tensor(y, dtype=torch.long)


# -------------------------------------------------------------
# Models (TCN)
# -------------------------------------------------------------
class SimpleTCN(nn.Module):
    """Old/simple TCN: 2-logit softmax head by default."""
    def __init__(self, in_dim=66, hidden=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)          # [B,66,T]
        h = self.net(x).squeeze(-1)    # [B,hidden]
        return self.head(h)            # [B,num_classes]


class ResTCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, p: float = 0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = self.drop(self.relu(self.bn(self.conv(x))))
        return x + y


class EnhancedTCN(nn.Module):
    """
    Matches checkpoints with keys:
      conv_in.0.weight, conv_in.1.running_mean, blocks.N.conv.weight, ..., head.weight
    Head may be 1-logit (BCE) or 2-logit (softmax).
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        kernel_in: int = 5,
        kernel_block: int = 3,
        dilations: List[int] | None = None,
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
            [ResTCNBlock(hidden, kernel_size=kernel_block, dilation=d, p=dropout) for d in dilations]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)     # [B,66,T]
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        h = x.mean(dim=-1)        # [B,hidden]
        return self.head(h)       # [B,out_dim]


# -------------------------------------------------------------
# Graph utilities (shared)
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Models (GCN family B: GRU-temporal)
# -------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor):
        x = torch.einsum("ij,btjc->btic", A_hat, x)
        return self.lin(x)


class GCNBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.2):
        super().__init__()
        self.gc = GraphConv(in_feats, out_feats)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor):
        return self.do(self.act(self.gc(x, A_hat)))


class GCNTemporalGRU(nn.Module):
    """
    GRU-temporal GCN: keys like g1.gc.lin.*, temporal.weight_ih_l0, head.*
    Head may be 1-logit or 2-logit.
    """
    def __init__(
        self,
        num_joints: int = 33,
        in_feats: int = 4,
        hidden: int = 64,
        dropout: float = 0.2,
        out_dim: int = 2,
    ):
        super().__init__()
        A = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A))
        self.g1 = GCNBlock(in_feats, hidden, dropout)
        self.g2 = GCNBlock(hidden, hidden, dropout)
        self.temporal = nn.GRU(input_size=num_joints * hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.g1(x, self.A_hat)
        x = self.g2(x, self.A_hat)
        b, t, v, c = x.shape
        x = x.reshape(b, t, v * c)
        _, h = self.temporal(x)
        h = h.squeeze(0)
        return self.head(h)


# -------------------------------------------------------------
# Models (GCN family A: Conv-temporal, matches models/train_gcn.py)
# -------------------------------------------------------------
class BNGraphBlock(nn.Module):
    """Matches train_gcn.py: block.lin + BN + ReLU + Dropout."""
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("vw,btwc->btvc", A_hat, x)   # [B,T,V,C]
        B, T, V, _ = x.shape
        x = self.lin(x)                               # [B,T,V,Cout]
        x = x.reshape(B * T * V, -1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = x.reshape(B, T, V, -1)
        return x


class ConvTemporalGCN(nn.Module):
    """
    Matches your train_gcn.py checkpoint:
      block1.*, block2.*, temporal.0.*, head.*
    Head can be 1-logit or 2-logit.
    """
    def __init__(
        self,
        num_joints: int,
        in_feats: int,
        gcn_hidden: int,
        gcn_out: int,
        tcn_hidden: int,
        out_dim: int,
        dropout: float = 0.3,
        kernel_size: int = 5,
    ):
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))

        self.block1 = BNGraphBlock(in_feats, gcn_hidden, dropout)
        self.block2 = BNGraphBlock(gcn_hidden, gcn_out, dropout)

        pad = (kernel_size - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(gcn_out, tcn_hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(tcn_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, self.A_hat)
        x = self.block2(x, self.A_hat)
        x = x.mean(dim=2)              # [B,T,gcn_out]
        x = x.permute(0, 2, 1)         # [B,gcn_out,T]
        h = self.temporal(x).squeeze(-1)
        return self.head(h)            # [B,out_dim]


# -------------------------------------------------------------
# Checkpoint -> model factory
# -------------------------------------------------------------
def _strip_module_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    if all(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _get_state_dict(ckpt: dict) -> dict:
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return _strip_module_prefix(ckpt["model"])
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return _strip_module_prefix(ckpt["state_dict"])
    raise KeyError("Checkpoint has no 'model' or 'state_dict' dict.")


def _infer_num_blocks(sd: dict) -> int:
    idxs: List[int] = []
    for k in sd.keys():
        if isinstance(k, str) and k.startswith("blocks.") and ".conv.weight" in k:
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0


def _infer_dilations(ckpt: dict, n_blocks: int) -> List[int]:
    d = ckpt.get("dilations", None)
    if isinstance(d, (list, tuple)) and len(d) == n_blocks:
        return [int(x) for x in d]
    return [2 ** i for i in range(n_blocks)] if n_blocks > 0 else [1, 2, 4]


def build_model_from_ckpt(ckpt: dict, arch: str) -> Tuple[nn.Module, str]:
    """
    Returns (model, prob_mode) where prob_mode is:
      - "sigmoid" for 1-logit binary head
      - "softmax" for 2-logit head
    """
    sd = _get_state_dict(ckpt)

    if arch == "gcn":
        # Family A: Conv-temporal GCN (train_gcn.py) has block1.*
        if any(k.startswith("block1.") for k in sd.keys()):
            # Infer dims from weights
            gcn_hidden = int(sd["block1.lin.weight"].shape[0])
            in_feats   = int(sd["block1.lin.weight"].shape[1])
            gcn_out    = int(sd["block2.lin.weight"].shape[0])

            tcn_hidden = int(sd["temporal.0.weight"].shape[0])
            kernel     = int(sd["temporal.0.weight"].shape[2])

            out_dim    = int(sd["head.weight"].shape[0])  # 1 or 2
            num_joints = int(ckpt.get("num_joints", 33))

            model = ConvTemporalGCN(
                num_joints=num_joints,
                in_feats=in_feats,
                gcn_hidden=gcn_hidden,
                gcn_out=gcn_out,
                tcn_hidden=tcn_hidden,
                out_dim=out_dim,
                dropout=float(ckpt.get("dropout", 0.3)),
                kernel_size=kernel,
            )
            prob_mode = "sigmoid" if out_dim == 1 else "softmax"
            return model, prob_mode

        # Family B: GRU-temporal GCN (g1/g2 + GRU)
        # Infer from weights if possible
        if "g1.gc.lin.weight" in sd:
            hidden = int(sd["g1.gc.lin.weight"].shape[0])
            in_feats = int(sd["g1.gc.lin.weight"].shape[1])
        else:
            hidden = int(ckpt.get("hidden", 64))
            in_feats = int(ckpt.get("in_feats", 4))

        out_dim = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 2
        num_joints = int(ckpt.get("num_joints", 33))

        model = GCNTemporalGRU(
            num_joints=num_joints,
            in_feats=in_feats,
            hidden=hidden,
            dropout=float(ckpt.get("dropout", 0.2)),
            out_dim=out_dim,
        )
        prob_mode = "sigmoid" if out_dim == 1 else "softmax"
        return model, prob_mode

    # arch == "tcn"
    # Enhanced TCN?
    if any(k.startswith("conv_in.") for k in sd.keys()) and any(k.startswith("blocks.") for k in sd.keys()):
        conv_w = sd["conv_in.0.weight"]         # [hidden, in_dim, k]
        hidden = int(conv_w.shape[0])
        in_dim = int(conv_w.shape[1])
        k_in   = int(conv_w.shape[2])

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

    # Simple TCN
    if "net.0.weight" in sd:
        in_dim = int(sd["net.0.weight"].shape[1])   # Conv1d weight: [hidden, in_dim, k]
        hidden = int(sd["net.0.weight"].shape[0])
    else:
        in_dim = int(ckpt.get("in_dim", 66))
        hidden = int(ckpt.get("hidden", 128))

    out_dim = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 2
    model = SimpleTCN(in_dim=in_dim, hidden=hidden, num_classes=out_dim, dropout=float(ckpt.get("dropout", 0.2)))
    prob_mode = "sigmoid" if out_dim == 1 else "softmax"
    return model, prob_mode


# -------------------------------------------------------------
# Predict probabilities
# -------------------------------------------------------------
@torch.no_grad()
def predict_probs(ckpt_path: str, loader: DataLoader, arch: str, device: str) -> Tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ckpt)

    model, prob_mode = build_model_from_ckpt(ckpt, arch=arch)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    probs_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)

        # logits can be [B,1], [B], or [B,2]
        if prob_mode == "sigmoid":
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            p = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs_all.append(p)
        y_all.append(yb.numpy())

    return np.concatenate(probs_all), np.concatenate(y_all)


# -------------------------------------------------------------
# Threshold sweep + OP selection
# -------------------------------------------------------------
def sweep_metrics(probs: np.ndarray, y_true: np.ndarray, thr: np.ndarray) -> Dict[str, np.ndarray]:
    P, R, F1, FPR = [], [], [], []
    for t in thr:
        pred = (probs >= t).astype(np.int32)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        P.append(pr); R.append(rc); F1.append(f1); FPR.append(fpr)
    return {"P": np.array(P), "R": np.array(R), "F1": np.array(F1), "FPR": np.array(FPR)}


def pick_ops(
    thr: np.ndarray,
    stats: Dict[str, np.ndarray],
    recall_floor_op1: float = 0.95,
    recall_floor_op3: float = 0.90,
):
    P, R, F1, FPR = stats["P"], stats["R"], stats["F1"], stats["FPR"]

    # OP2: max F1
    i2 = int(np.argmax(F1))
    op2 = dict(thr=float(thr[i2]), f1=float(F1[i2]), precision=float(P[i2]), recall=float(R[i2]), fpr=float(FPR[i2]))

    # OP1: MIN threshold s.t. recall >= floor (tie-break: higher precision)
    mask1 = R >= recall_floor_op1
    if mask1.any():
        idxs = np.where(mask1)[0]
        min_thr = thr[idxs].min()
        cands = idxs[thr[idxs] == min_thr]
        i1 = int(cands[np.argmax(P[cands])])
    else:
        i1 = int(np.argmax(R + 1e-6 * P))
    op1 = dict(thr=float(thr[i1]), recall_at_thr=float(R[i1]), precision=float(P[i1]), fpr=float(FPR[i1]))

    # OP3: MIN FPR s.t. recall >= floor (tie-break: higher precision)
    mask3 = R >= recall_floor_op3
    if mask3.any():
        idxs = np.where(mask3)[0]
        best_fpr = FPR[idxs].min()
        cands = idxs[FPR[idxs] == best_fpr]
        i3 = int(cands[np.argmax(P[cands])])
    else:
        i3 = int(np.argmin(FPR))
    op3 = dict(thr=float(thr[i3]), fpr=float(FPR[i3]), recall_at_thr=float(R[i3]), precision=float(P[i3]))

    return op1, op2, op3


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["tcn", "gcn"], default="tcn")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps_default", type=float, default=30.0, help="GCN only: fallback FPS if window NPZ lacks fps.")
    ap.add_argument("--recall_floor_op1", type=float, default=0.95)
    ap.add_argument("--recall_floor_op3", type=float, default=0.90)
    args = ap.parse_args()

    if args.arch == "tcn":
        ds = WindowNPZ(args.val_dir, skip_unlabeled=True)
    else:
        ds = WindowNPZGraph(args.val_dir, skip_unlabeled=True, fps_default=args.fps_default)

    if len(ds) == 0:
        raise SystemExit(f"No labelled windows found under {args.val_dir}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    probs, y_true = predict_probs(args.ckpt, loader, arch=args.arch, device=args.device)

    thr = np.linspace(0.01, 0.99, 99)
    stats = sweep_metrics(probs, y_true, thr)
    op1, op2, op3 = pick_ops(thr, stats, recall_floor_op1=args.recall_floor_op1, recall_floor_op3=args.recall_floor_op3)

    out = {
        "arch": args.arch,
        "val_dir": str(args.val_dir),
        "ckpt": str(args.ckpt),
        "floors": {"op1_recall": float(args.recall_floor_op1), "op3_recall": float(args.recall_floor_op3)},
        "OP1_high_recall": op1,
        "OP2_balanced": op2,
        "OP3_low_alarm": op3,
    }

    # tiny YAML writer (no external deps)
    def _yaml_dump(d, indent=0):
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                lines.extend(_yaml_dump(v, indent + 2))
            else:
                lines.append(" " * indent + f"{k}: {v}")
        return lines

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(_yaml_dump(out)) + "\n")

    print(f"[OK] wrote ops → {args.out}")
    print(f"  OP1 thr={op1['thr']:.3f} (recall={op1['recall_at_thr']:.3f})")
    print(f"  OP2 thr={op2['thr']:.3f} (f1={op2['f1']:.3f})")
    print(f"  OP3 thr={op3['thr']:.3f} (fpr={op3['fpr']:.4f}, recall={op3['recall_at_thr']:.3f})")


if __name__ == "__main__":
    main()
