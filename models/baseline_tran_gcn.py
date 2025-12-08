
"""
Train a simple spatio-temporal GCN on windowed pose sequences.

Input windows are the same NPZ files used for the TCN:
  - xy  : [T, 33, 2]  (normalized MediaPipe Pose 2D joints)
  - conf: [T, 33]     (visibility / confidence)
  - y   : scalar label (0/1) or string ('adl' / 'fall')

Model:
  - Graph layers over the 33-joint skeleton for each time step
  - Temporal 1D conv over the sequence of frame embeddings
  - Final linear → fall / no-fall logit

Usage (example):

  python models/train_gcn.py \
    --train_dir data/processed/le2i/windows_W48_S12/train \
    --val_dir   data/processed/le2i/windows_W48_S12/val \
    --epochs 50 --batch 64 --lr 1e-3 --seed 33724876 \
    --save_dir outputs/le2i_gcn_W48S12
"""

import os, glob, argparse, random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
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
def _label_from_npz(d) -> float | None:
    """
    Try to read a label from an NPZ window.
    Returns:
      1.0 for fall, 0.0 for adl/negative,
      None for unlabeled (e.g. label == -1 or missing).
    """
    if "y" in d.files:
        y = d["y"]
        if isinstance(y, np.ndarray):
            if y.shape == ():
                y = y.item()
            else:
                y = y.ravel()[0]
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

            # numeric?
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

    return None  # completely unlabeled


# -------------------------------------------------------------------
# Dataset: keep full [T, 33, 2] geometry for GCN
# -------------------------------------------------------------------
class WindowNPZGraph(Dataset):
    """
    NPZ windows for GCN:

      - xy   : [T, 33, 2]
      - conf : [T, 33]
      - y    : scalar (0/1) or label string

    Returns:
      x: [T, 33, 2]  (float32, confidence-gated, NaNs→0)
      y: [1]         (float32 0./1.)
    """
    def __init__(self, root: str, skip_unlabeled: bool = True):
        all_files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not all_files:
            raise FileNotFoundError(f"No .npz windows found under: {root}")

        kept, skipped = [], 0
        for p in all_files:
            try:
                d = np.load(p, allow_pickle=False)
                y = _label_from_npz(d)
                if skip_unlabeled and y is None:
                    skipped += 1
                    continue
                kept.append(p)
            except Exception:
                skipped += 1

        if not kept:
            raise FileNotFoundError(f"All windows under {root} were unlabeled or unreadable.")
        if skipped:
            print(f"[WindowNPZGraph] skipped {skipped} unlabeled/unreadable windows under {root}")
        self.files = kept

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=False)
        xy = d["xy"].astype(np.float32)      # [T,33,2]
        conf = d["conf"].astype(np.float32)  # [T,33]

        # NaNs → 0, gate by confidence
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        x = xy * conf[..., None]             # [T,33,2]

        y = _label_from_npz(d)
        if y is None:
            y = 0.0

        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32)


# -------------------------------------------------------------------
# Skeleton graph (33-joint MediaPipe Pose, approximated)
# -------------------------------------------------------------------
def build_mediapipe_adjacency(num_joints: int = 33) -> np.ndarray:
    """
    Undirected adjacency for MediaPipe Pose (33 landmarks).
    This is a reasonable subset of the official connections;
    it does not need to be perfect for the model to work.
    """
    edges = [
        # torso & head
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),

        # left arm
        (11, 13), (13, 15), (15, 17),
        # right arm
        (12, 14), (14, 16), (16, 18),

        # left leg
        (23, 25), (25, 27), (27, 29), (29, 31),
        # right leg
        (24, 26), (26, 28), (28, 30), (30, 32),

        # cross-links
        (7, 28), (8, 27),
    ]

    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        if 0 <= i < num_joints and 0 <= j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # self-connections
    for i in range(num_joints):
        A[i, i] = 1.0
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Standard symmetric normalisation: D^(-1/2) A D^(-1/2)
    """
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt


# -------------------------------------------------------------------
# Simple spatio-temporal GCN
# -------------------------------------------------------------------
class GCNTemporal(nn.Module):
    """
    Very small spatio-temporal GCN:

      1. Graph aggregation over 33 joints at each time step (using A_hat).
      2. Two point-wise MLP layers on joint features.
      3. Global mean over joints → frame embeddings [B,T,F].
      4. Temporal Conv1d over the sequence → pooled → Linear → logit.

    Input:  x [B, T, 33, 2]
    Output: logits [B,1]
    """
    def __init__(
        self,
        num_joints: int = 33,
        in_feats: int = 2,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        tcn_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        A = build_mediapipe_adjacency(num_joints)
        A_hat = normalize_adjacency(A)
        self.register_buffer("A_hat", torch.from_numpy(A_hat))  # [V,V]

        self.gc1 = nn.Linear(in_feats, gcn_hidden)
        self.gc2 = nn.Linear(gcn_hidden, gcn_out)
        self.relu = nn.ReLU()

        self.temporal = nn.Sequential(
            nn.Conv1d(gcn_out, tcn_hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C]  (C=2, V=33)
        B, T, V, C = x.shape
        A_hat = self.A_hat  # [V,V]

        # Graph aggregation: for each frame, aggregate neighbours
        # x_agg[b,t,v,c] = sum_w A_hat[v,w] * x[b,t,w,c]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)  # [B,T,V,C]

        # First "GCN" layer (pointwise over features)
        x = self.relu(self.gc1(x))  # [B,T,V,gcn_hidden]

        # Second graph aggregation + MLP
        x = torch.einsum("vw,btwc->btvc", A_hat, x)  # [B,T,V,gcn_hidden]
        x = self.relu(self.gc2(x))                  # [B,T,V,gcn_out]

        # Pool over joints → frame embeddings
        x = x.mean(dim=2)   # [B,T,gcn_out]

        # Temporal conv expects [B,C,T]
        x = x.permute(0, 2, 1)  # [B,gcn_out,T]
        x = self.temporal(x).squeeze(-1)  # [B,tcn_hidden]

        logits = self.head(x)  # [B,1]
        return logits


# -------------------------------------------------------------------
# Validation with threshold sweep (like TCN)
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate_with_sweep(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_y = [], []
    batch0_printed = False

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if not batch0_printed:
            print(f"[val] batch0 positives = {int(yb.sum().item())}/{yb.numel()}")
            batch0_printed = True
        logits = model(xb).squeeze(-1)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    if not all_logits:
        return dict(P=0.0, R=0.0, F1=0.0, thr=0.5, note="empty val loader")

    y_true = np.concatenate(all_y, axis=0).ravel().astype(int)
    if y_true.sum() == 0:
        return dict(P=0.0, R=0.0, F1=0.0, thr=0.5, note="val has no positives")

    probs = torch.sigmoid(
        torch.from_numpy(np.concatenate(all_logits, axis=0).ravel())
    ).numpy()

    best = dict(F1=-1.0, P=0.0, R=0.0, thr=0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best["F1"]:
            best.update(dict(F1=float(f1), P=float(pr), R=float(rc), thr=float(thr)))
    return best


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a small GCN+TCN on skeleton windows.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    train_ds = WindowNPZGraph(args.train_dir, skip_unlabeled=True)
    val_ds = WindowNPZGraph(args.val_dir, skip_unlabeled=True)

    # Inspect one sample
    x0, y0 = train_ds[0]
    T, V, C = x0.shape
    print(f"[info] window shape (T={T}, V={V}, C={C}); first y={float(y0.item())}")

    device = pick_device()
    print(f"[info] device: {device.type}")

    model = GCNTemporal(num_joints=V, in_feats=C).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train GCN ep{ep}", leave=False)
        batch0_printed = False

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            if not batch0_printed:
                print(f"[train] epoch {ep} batch0 positives = {int(yb.sum().item())}/{yb.numel()}")
                batch0_printed = True

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        stats = evaluate_with_sweep(model, val_loader, device)
        note = stats.get("note", "")
        print(f"[val] P={stats['P']:.3f} R={stats['R']:.3f} F1={stats['F1']:.3f} @thr={stats['thr']:.2f} {note}")

        if stats["F1"] >= best_f1:
            best_f1 = stats["F1"]
            torch.save(
                {"model": model.state_dict(), "num_joints": V, "in_feats": C, "best_thr": stats["thr"]},
                best_path,
            )
            print(f"[save] {best_path} (F1={best_f1:.3f} @thr={stats['thr']:.2f})")

    print(f"[done] best F1={best_f1:.3f}  ckpt={best_path}")


if __name__ == "__main__":
    main()
