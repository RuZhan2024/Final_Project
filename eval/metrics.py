
"""
Evaluate a trained model on a directory of window NPZ files and write a JSON report.

Supports:
  - TCN windows : x [T,66]
  - GCN windows : x [T,33,4] (uses fps stored in each window NPZ for velocity)

If an ops YAML is provided (from fit_ops.py), we report metrics at OP1/OP2/OP3.
Otherwise, we still produce the threshold sweep arrays for plotting.

Important (checkpoint compatibility)
-----------------------------------
- TCN checkpoints may be:
    (A) simple TCN with keys like net.* and usually 2-logit head (softmax), OR
    (B) enhanced TCN with keys like conv_in.*, blocks.*, head.* and often 1-logit head (sigmoid).
- GCN checkpoints may be:
    (A) conv-temporal GCN from your models/train_gcn.py:
        keys like block1.*, block2.*, temporal.0.*, head.* (often 1-logit -> sigmoid), OR
    (B) GRU-temporal GCN:
        keys like g1.*, g2.*, temporal.weight_ih_l0, head.* (1-logit sigmoid or 2-logit softmax).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# -------------------------------------------------------------
# YAML (tiny) reader: supports the simple structure written by fit_ops.py
# -------------------------------------------------------------
def load_simple_yaml(path: str) -> Dict[str, Any]:
    # If PyYAML is present, prefer it.
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.strip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            key, _, val = raw.strip().partition(":")
            val = val.strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]
            if val == "":
                cur[key] = {}
                stack.append((indent + 2, cur[key]))
            else:
                if val.lower() in ("true", "false"):
                    cur[key] = (val.lower() == "true")
                else:
                    try:
                        if "." in val:
                            cur[key] = float(val)
                        else:
                            cur[key] = int(val)
                    except Exception:
                        cur[key] = val
    return root


# -------------------------------------------------------------
# Datasets
# -------------------------------------------------------------
def _label_from_npz(d: np.lib.npyio.NpzFile) -> int:
    """
    Returns:
      0/1 for labelled
      -1 for unlabeled
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
        try:
            v = float(lab)
            if v < 0:
                return -1
            return 1 if v >= 0.5 else 0
        except Exception:
            s = str(lab).lower()
            if s in ("fall", "1", "true", "pos", "positive"):
                return 1
            if s in ("adl", "0", "false", "neg", "negative", "normal", "nofall", "no_fall", "nonfall"):
                return 0
            return -1

    return -1


class WindowNPZ(Dataset):
    """TCN windows: returns (x[T,66], y, meta)."""
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
            xy = d["xy"].astype(np.float32)
            conf = d["conf"].astype(np.float32)
            y = _label_from_npz(d)

            meta = {
                "path": p,
                "video_id": str(d["video_id"]) if "video_id" in d.files else None,
                "start": int(d["start"]) if "start" in d.files else None,
                "end": int(d["end"]) if "end" in d.files else None,
                "fps": float(d["fps"]) if "fps" in d.files else None,
            }

        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        x = (xy * conf[..., None]).reshape(xy.shape[0], -1)  # [T,66]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta


class WindowNPZGraph(Dataset):
    """GCN windows: returns (x[T,33,4], y, meta)."""
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
            xy = d["xy"].astype(np.float32)
            conf = d["conf"].astype(np.float32)
            y = _label_from_npz(d)
            fps = float(d["fps"]) if "fps" in d.files else self.fps_default

            meta = {
                "path": p,
                "video_id": str(d["video_id"]) if "video_id" in d.files else None,
                "start": int(d["start"]) if "start" in d.files else None,
                "end": int(d["end"]) if "end" in d.files else None,
                "fps": float(fps),
            }

        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        x = xy * conf[..., None]        # [T,33,2]
        pelvis = x[:, 23:24, :]         # [T,1,2]
        x_rel = x - pelvis              # [T,33,2]

        vel = np.zeros_like(x_rel)
        vel[1:] = (x_rel[1:] - x_rel[:-1]) * fps

        feats = np.concatenate([x_rel, vel], axis=-1)  # [T,33,4]
        return torch.from_numpy(feats), torch.tensor(y, dtype=torch.long), meta


def collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


# -------------------------------------------------------------
# Models (TCN family)
# -------------------------------------------------------------
class SimpleTCN(nn.Module):
    def __init__(self, in_ch=66, hidden=128, dropout=0.2, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,C,T]
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

    def forward(self, x):
        y = self.drop(self.relu(self.bn(self.conv(x))))
        return x + y


class EnhancedTCN(nn.Module):
    """
    Matches keys:
      conv_in.0.weight, conv_in.1.running_mean, blocks.N.conv.weight, ..., head.weight
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
        x = x.transpose(1, 2)  # [B,C,T]
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        h = x.mean(dim=-1)
        return self.head(h)


# -------------------------------------------------------------
# GCN utilities + models (two checkpoint styles)
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


# --- GCN style B: GRU-temporal (g1/g2 + GRU) ---
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
        return self.head(h.squeeze(0))


# --- GCN style A: conv-temporal (block1/block2 + temporal.0 conv) ---
class BNGraphBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("vw,btwc->btvc", A_hat, x)  # [B,T,V,C]
        B, T, V, _ = x.shape
        x = self.lin(x)                              # [B,T,V,Cout]
        x = x.reshape(B * T * V, -1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x.reshape(B, T, V, -1)


class ConvTemporalGCN(nn.Module):
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
    Returns (model, prob_mode):
      - "sigmoid" for 1-logit binary head
      - "softmax" for 2-logit head
    """
    sd = _get_state_dict(ckpt)

    if arch == "gcn":
        # style A: conv-temporal GCN (train_gcn.py) has block1.*
        if any(k.startswith("block1.") for k in sd.keys()):
            gcn_hidden = int(sd["block1.lin.weight"].shape[0])
            in_feats = int(sd["block1.lin.weight"].shape[1])
            gcn_out = int(sd["block2.lin.weight"].shape[0])

            tcn_hidden = int(sd["temporal.0.weight"].shape[0])
            kernel = int(sd["temporal.0.weight"].shape[2])

            out_dim = int(sd["head.weight"].shape[0])  # 1 or 2
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
            return model, ("sigmoid" if out_dim == 1 else "softmax")

        # style B: GRU-temporal GCN (g1/g2 + GRU)
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
        return model, ("sigmoid" if out_dim == 1 else "softmax")

    # arch == "tcn"
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
        return model, ("sigmoid" if out_dim == 1 else "softmax")

    # simple TCN (net.*)
    if "net.0.weight" in sd:
        in_ch = int(sd["net.0.weight"].shape[1])
        hidden = int(sd["net.0.weight"].shape[0])
    else:
        in_ch = int(ckpt.get("in_ch", 66))
        hidden = int(ckpt.get("hidden", 128))

    out_dim = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 2
    model = SimpleTCN(in_ch=in_ch, hidden=hidden, dropout=float(ckpt.get("dropout", 0.2)), out_dim=out_dim)
    return model, ("sigmoid" if out_dim == 1 else "softmax")


# -------------------------------------------------------------
# evaluation
# -------------------------------------------------------------
def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_probs(ckpt_path: str, ds: Dataset, arch: str, batch=256):
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ckpt)

    model, prob_mode = build_model_from_ckpt(ckpt, arch=arch)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_keep_meta)

    probs, ys, metas = [], [], []
    for xb, yb, mb in loader:
        xb = xb.to(device)
        logits = model(xb)

        if prob_mode == "sigmoid":
            # logits: [B,1] or [B]
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            p = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            # logits: [B,2]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs.append(p)
        ys.append(yb.numpy())
        metas.extend(mb)

    return np.concatenate(probs), np.concatenate(ys), metas


def sweep(probs: np.ndarray, y_true: np.ndarray, thr: np.ndarray):
    P, R, F1, FPR = [], [], [], []
    for t in thr:
        pred = (probs >= t).astype(np.int32)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        P.append(pr); R.append(rc); F1.append(f1); FPR.append(fpr)
    return np.array(P), np.array(R), np.array(F1), np.array(FPR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ops", default=None, help="ops YAML from fit_ops.py")
    ap.add_argument("--arch", choices=["tcn", "gcn"], default="tcn")
    ap.add_argument("--fps", type=float, default=None, help="Unused (kept for CLI compatibility).")
    ap.add_argument("--fps_default", type=float, default=30.0, help="GCN only: fallback fps if window NPZ lacks fps")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--report", required=True)
    ap.add_argument("--report_dataset_name", default="test")
    args = ap.parse_args()

    if args.arch == "tcn":
        ds = WindowNPZ(args.eval_dir, skip_unlabeled=True)
    else:
        ds = WindowNPZGraph(args.eval_dir, skip_unlabeled=True, fps_default=args.fps_default)

    if len(ds) == 0:
        raise SystemExit(f"No labelled windows under {args.eval_dir}")

    probs, y_true, metas = predict_probs(args.ckpt, ds, arch=args.arch, batch=args.batch)

    thr = np.linspace(0.01, 0.99, 99)
    P, R, F1, FPR = sweep(probs, y_true, thr)

    # Load ops thresholds if provided
    thr1 = thr2 = thr3 = None
    if args.ops:
        ops = load_simple_yaml(args.ops)
        try:
            thr1 = float(ops["OP1_high_recall"]["thr"])
            thr2 = float(ops["OP2_balanced"]["thr"])
            thr3 = float(ops["OP3_low_alarm"]["thr"])
        except Exception:
            pass

    def at_thr(t):
        pred = (probs >= t).astype(np.int32)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(pr), float(rc), float(f1), float(fpr)

    ops_report = {}
    if thr1 is not None:
        p1, r1, f1v, fpr1 = at_thr(thr1)
        ops_report["OP1_high_recall"] = {"thr": thr1, "precision": p1, "recall": r1, "f1": f1v, "fpr": fpr1}
    if thr2 is not None:
        p2, r2, f2v, fpr2 = at_thr(thr2)
        ops_report["OP2_balanced"] = {"thr": thr2, "precision": p2, "recall": r2, "f1": f2v, "fpr": fpr2}
    if thr3 is not None:
        p3, r3, f3v, fpr3 = at_thr(thr3)
        ops_report["OP3_low_alarm"] = {"thr": thr3, "precision": p3, "recall": r3, "f1": f3v, "fpr": fpr3}

    have_meta = any((m.get("start") is not None and m.get("end") is not None) for m in metas)

    report = {
        "arch": args.arch,
        "dataset": args.report_dataset_name,
        "eval_dir": str(args.eval_dir),
        "ckpt": str(args.ckpt),
        "n_windows": int(len(ds)),
        "sweep": {
            "thr": thr.tolist(),
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
            "fpr": FPR.tolist(),
        },
        "ops": ops_report,
    }
    if not have_meta:
        report["note"] = "Window metadata missing start/end; FA/24h not computed here (FPR is reported)."

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[report]", args.report)


if __name__ == "__main__":
    main()
