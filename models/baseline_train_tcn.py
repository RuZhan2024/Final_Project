#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# -------------------------
# Utilities / Reproducible
# -------------------------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -------------------------
# Label parsing helpers
# -------------------------
def _label_from_npz(d) -> float | None:
    """
    Try to read a label from an NPZ window.
    Returns:
      1.0 for fall, 0.0 for adl/negative,
      None for unlabeled (e.g., label == -1 or missing).
    Priority: numeric 'y' -> string/numeric 'label'/'y_label'/'target'.
    """
    # Prefer numeric 'y' (0/1)
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

    # Fallback to string/numeric 'label'
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
            # unknown string → unlabeled
            return None

    # no label fields at all → unlabeled
    return None

# -------------------------
# Dataset
# -------------------------
class WindowNPZ(Dataset):
    """
    Expects per-window NPZ with:
      - xy   : [W, 33, 2] float32
      - conf : [W, 33]    float32
      - y    : scalar (0/1) OR 'label' string ('adl'/'fall') OR 'label'=-1 for unlabeled
    Produces:
      - x: [W, 33*2]  (xy gated by conf, NaNs->0)
      - y: [1]        float(0./1.)
    By default, unlabeled windows are skipped.
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
                # if unreadable, skip
                skipped += 1

        if not kept:
            raise FileNotFoundError(f"All windows under {root} were unlabeled or unreadable.")
        if skipped:
            print(f"[WindowNPZ] skipped {skipped} unlabeled/unreadable windows under {root}")
        self.files = kept

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=False)
        xy  = d["xy"].astype(np.float32)   # [W,33,2]
        conf= d["conf"].astype(np.float32) # [W,33]
        xy  = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        x = xy * conf[..., None]           # [W,33,2]
        x = x.reshape(x.shape[0], -1)      # [W, 33*2]
        y = _label_from_npz(d)
        if y is None:
            # Should not happen if skip_unlabeled=True, but guard anyway
            y = 0.0
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32)

# -------------------------
# Model (Tiny TCN)
# -------------------------
class TCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(hid, hid, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, C, T]
        h = self.net(x).squeeze(-1)
        return self.head(h)     # [B, 1] (logits)

# -------------------------
# Metrics
# -------------------------
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

    probs = torch.sigmoid(torch.from_numpy(np.concatenate(all_logits, axis=0).ravel())).numpy()

    best = dict(F1=-1.0, P=0.0, R=0.0, thr=0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        if f1 > best["F1"]:
            best.update(dict(F1=float(f1), P=float(pr), R=float(rc), thr=float(thr)))
    return best

# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a tiny TCN on windowed pose sequences.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir",   required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=33724876)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets (skip unlabeled by default)
    train_ds = WindowNPZ(args.train_dir, skip_unlabeled=True)
    val_ds   = WindowNPZ(args.val_dir,   skip_unlabeled=True)

    # Input dims
    x0, y0 = train_ds[0]
    T, C = x0.shape
    print(f"[info] window shape (T={T}, C={C}); first y={float(y0.item())}")

    # Device
    device = pick_device()
    print(f"[info] device: {device.type}")

    # Model / Optim / Loss
    model = TCN(in_ch=C).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train ep{ep}", leave=False)
        batch0_printed = False

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            if not batch0_printed:
                print(f"[train] epoch {ep} batch0 positives = {int(yb.sum().item())}/{yb.numel()}")
                batch0_printed = True

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        stats = evaluate_with_sweep(model, val_loader, device)
        note  = stats.get("note", "")
        print(f"val: P={stats['P']:.3f} R={stats['R']:.3f} F1={stats['F1']:.3f} @thr={stats['thr']:.2f} {note}")

        if stats["F1"] >= best_f1:
            best_f1 = stats["F1"]
            torch.save({"model": model.state_dict(), "in_ch": C, "best_thr": stats["thr"]}, best_path)
            print(f"[save] {best_path} (F1={best_f1:.3f} @thr={stats['thr']:.2f})")

    print(f"[done] best F1={best_f1:.3f}  ckpt={best_path}")

if __name__ == "__main__":
    main()
