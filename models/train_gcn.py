
# -*- coding: utf-8 -*-
"""
Train a spatio-temporal GCN on windowed pose sequences (MediaPipe Pose, 33 joints).

Input windows are NPZ files:
  - xy  : [T, 33, 2]
  - conf: [T, 33]
  - y   : scalar label (0/1) or string ('adl' / 'fall') depending on your pipeline

GCN features:
  - pelvis-centred (x,y)
  - per-second velocity (dx/dt, dy/dt) using fps from each window NPZ (fallback fps_default)

Output:
  - best.pt in save_dir
  - optional JSON test report with OP1/OP2/OP3 (precision/recall/f1/fpr vs thresholds)
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


# -------------------------------------------------------------------
# Repro / device helpers
# -------------------------------------------------------------------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------------
# Label helper (same semantics as train_tcn.py)
# -------------------------------------------------------------------
def _label_from_npz(d) -> Optional[float]:
    """
    Returns:
      1.0 for fall, 0.0 for adl/negative,
      None for unlabeled (e.g. label == -1 or missing).
    """
    if "y" in d.files:
        y = d["y"]
        if isinstance(y, np.ndarray):
            y = y.item() if y.shape == () else y.ravel()[0]
        try:
            y = float(y)
        except Exception:
            return None
        if y < 0:
            return None
        return 1.0 if y >= 0.5 else 0.0

    for k in ("label", "y_label", "target"):
        if k in d.files:
            lab = d[k]
            if isinstance(lab, bytes):
                lab = lab.decode()
            elif isinstance(lab, np.ndarray):
                try:
                    lab = lab.item()
                except Exception:
                    lab = str(lab)

            try:
                v = float(lab)
                if v < 0:
                    return None
                return 1.0 if v >= 0.5 else 0.0
            except Exception:
                pass

            s = str(lab).lower()
            if s in ("fall", "1", "true", "pos", "positive"):
                return 1.0
            if s in ("adl", "0", "false", "neg", "negative", "normal", "nofall", "no_fall"):
                return 0.0
            return None

    return None


# -------------------------------------------------------------------
# Dataset: keep full [T, 33, 2] geometry for GCN (+ velocity)
# -------------------------------------------------------------------
class WindowNPZGraph(Dataset):
    """
    Returns:
      x: [T,33,4] float32
      y: [1] float32 (0/1)  (ready for BCEWithLogitsLoss with logits [B,1])
    """
    def __init__(self, root: str, skip_unlabeled: bool = True, fps_default: float = 30.0):
        self.files = sorted([str(p) for p in Path(root).glob("*.npz")])
        kept: List[str] = []
        for p in self.files:
            with np.load(p, allow_pickle=False) as d:
                y = _label_from_npz(d)
            if (not skip_unlabeled) or (y in (0.0, 1.0)):
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

        if y is None:
            # Should not happen if skip_unlabeled=True, but keep safe.
            y = 0.0

        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        x = xy * conf[..., None]                # [T,33,2]
        pelvis = x[:, 23:24, :]                 # [T,1,2]
        x_rel = x - pelvis                      # [T,33,2]

        vel = np.zeros_like(x_rel)
        vel[1:] = (x_rel[1:] - x_rel[:-1]) * fps
        feats = np.concatenate([x_rel, vel], axis=-1)  # [T,33,4]

        x_t = torch.from_numpy(feats)                       # [T,33,4]
        y_t = torch.tensor([float(y)], dtype=torch.float32)  # [1]
        return x_t, y_t


# -------------------------------------------------------------------
# Skeleton graph (33-joint MediaPipe Pose, approximated)
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# GCN block + spatio-temporal model
# -------------------------------------------------------------------
class GCNBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,Cin]  A_hat: [V,V]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)
        B, T, V, _ = x.shape
        x = self.lin(x)                      # [B,T,V,Cout]
        x = x.reshape(B * T * V, -1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(B, T, V, -1)
        return x


class GCNTemporal(nn.Module):
    """
    Input:  x [B,T,V,C]
    Output: logits [B,1]
    """
    def __init__(
        self,
        num_joints: int = 33,
        in_feats: int = 4,
        gcn_hidden: int = 64,
        gcn_out: int = 64,
        tcn_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        A_hat = normalize_adjacency(build_mediapipe_adjacency(num_joints))
        self.register_buffer("A_hat", torch.from_numpy(A_hat))  # [V,V]

        self.block1 = GCNBlock(in_feats, gcn_hidden, dropout)
        self.block2 = GCNBlock(gcn_hidden, gcn_out, dropout)

        self.temporal = nn.Sequential(
            nn.Conv1d(gcn_out, tcn_hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, self.A_hat)
        h = self.block2(h, self.A_hat)
        h = h.mean(dim=2)            # pool joints -> [B,T,F]
        h = h.permute(0, 2, 1)       # [B,F,T]
        h = self.temporal(h).squeeze(-1)
        return self.head(h)          # [B,1]


# -------------------------------------------------------------------
# Metrics helpers
# -------------------------------------------------------------------
def _fpr_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


@torch.no_grad()
def evaluate_with_sweep(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_logits, all_y = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)            # [B]
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())   # [B,1]

    if not all_logits:
        return dict(P=0.0, R=0.0, F1=0.0, thr=0.5, note="empty val loader")

    y_true = np.concatenate(all_y, axis=0).reshape(-1).astype(int)
    if y_true.size == 0:
        return dict(P=0.0, R=0.0, F1=0.0, thr=0.5, note="no labels")

    probs = torch.sigmoid(torch.from_numpy(np.concatenate(all_logits, axis=0).reshape(-1))).numpy()

    best = dict(F1=-1.0, P=0.0, R=0.0, thr=0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best["F1"]:
            best.update(dict(F1=float(f1), P=float(pr), R=float(rc), thr=float(thr)))
    return best


@torch.no_grad()
def collect_probs_and_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)            # [B]
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())   # [B,1]
    y_true = np.concatenate(all_y, axis=0).reshape(-1).astype(int)
    probs = torch.sigmoid(torch.from_numpy(np.concatenate(all_logits, axis=0).reshape(-1))).numpy()
    return probs, y_true


def compute_ops_from_sweep(probs: np.ndarray, y_true: np.ndarray, recall_floor: float = 0.80) -> Dict[str, Dict[str, float]]:
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        fpr = _fpr_from_preds(y_true, pred)
        rows.append(
            dict(thr=float(thr), precision=float(pr), recall=float(rc), f1=float(f1), fpr=float(fpr))
        )

    op2 = max(rows, key=lambda r: r["f1"])
    op1 = max(rows, key=lambda r: (r["recall"], r["f1"]))

    eligible = [r for r in rows if r["recall"] >= recall_floor]
    if not eligible:
        eligible = rows
    op3 = min(eligible, key=lambda r: (r["fpr"], -r["f1"]))

    return {"OP1_high_recall": op1, "OP2_balanced": op2, "OP3_low_alarm": op3}


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a GCN+temporal head on skeleton windows (MediaPipe Pose).")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", required=False)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--report_json", default=None)
    ap.add_argument("--report_dataset_name", default="test")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)

    train_ds = WindowNPZGraph(args.train_dir, skip_unlabeled=True, fps_default=args.fps_default)
    val_ds   = WindowNPZGraph(args.val_dir,   skip_unlabeled=True, fps_default=args.fps_default)

    if len(train_ds) == 0:
        raise SystemExit(f"[ERR] No labelled windows found in train_dir={args.train_dir}")
    if len(val_ds) == 0:
        raise SystemExit(f"[ERR] No labelled windows found in val_dir={args.val_dir}")

    x0, y0 = train_ds[0]
    T, V, C = x0.shape
    print(f"[info] window shape: T={T}, V={V}, C={C}; first y={y0.item():.0f}")

    device = pick_device()
    print(f"[info] device: {device.type}")

    model = GCNTemporal(num_joints=V, in_feats=C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train GCN ep{ep}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)               # [B,1] float32

            opt.zero_grad(set_to_none=True)
            logits = model(xb)               # [B,1]
            loss = criterion(logits, yb)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        stats = evaluate_with_sweep(model, val_loader, device)
        print(f"[val] P={stats['P']:.3f} R={stats['R']:.3f} F1={stats['F1']:.3f} @thr={stats['thr']:.2f} {stats.get('note','')}")

        if stats["F1"] >= best_f1:
            best_f1 = stats["F1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_joints": V,
                    "in_feats": C,
                    "best_thr": stats["thr"],
                },
                best_path,
            )
            print(f"[save] {best_path} (F1={best_f1:.3f} @thr={stats['thr']:.2f})")

    print(f"[done] best F1={best_f1:.3f}  ckpt={best_path}")

    # Optional: test-set report (load best checkpoint)
    if args.test_dir and args.report_json:
        print(f"[report] evaluating best checkpoint on test_dir={args.test_dir}")

        ck = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        test_ds = WindowNPZGraph(args.test_dir, skip_unlabeled=True, fps_default=args.fps_default)
        if len(test_ds) == 0:
            print("[report] no labelled windows in test_dir; skipping JSON report.")
            return

        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        probs, y_true = collect_probs_and_labels(model, test_loader, device)

        ops = compute_ops_from_sweep(probs, y_true)
        report = {
            "arch": "gcn",
            "dataset": args.report_dataset_name,
            "n_windows": int(len(test_ds)),
            "pos_windows": int(y_true.sum()),
            "ops": ops,
        }

        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report_json}")


if __name__ == "__main__":
    main()
