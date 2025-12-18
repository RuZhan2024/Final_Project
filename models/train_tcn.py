
# -*- coding: utf-8 -*-

import os, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
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
# Dataset (same x as before)
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

        # NaNs / inf → 0, then gate by confidence
        xy  = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        x = xy * conf[..., None]           # [W,33,2]
        x = x.reshape(x.shape[0], -1)      # [W, 33*2]
        y = _label_from_npz(d)
        if y is None:
            # Should not happen if skip_unlabeled=True, but guard anyway
            y = 0.0
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32)


# -------------------------
# Enhanced TCN model
# -------------------------
class ResTCNBlock(nn.Module):
    """
    Residual temporal block with optional dilation:
      x -> Conv1d -> BN -> ReLU -> Dropout -> +x
    """
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, p: float = 0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            ch, ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn   = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        return x + out


class TCN(nn.Module):
    """
    Enhanced TCN:

      - input:  [B, T, C]  (C = 33*2 features per frame)
      - conv_in + BN + ReLU
      - 3 residual TCN blocks with dilations 1, 2, 4
      - global average pooling over time
      - linear head → fall logit

    Signature kept the same as original TCN(in_ch, hid=128, p=0.2)
    so that server/app.py can still import and use it.
    """
    def __init__(self, in_ch: int, hid: int = 128, p: float = 0.3):
        super().__init__()

        self.conv_in = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            ResTCNBlock(hid, kernel_size=3, dilation=1, p=p),
            ResTCNBlock(hid, kernel_size=3, dilation=2, p=p),
            ResTCNBlock(hid, kernel_size=3, dilation=4, p=p),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → [B, C, T]
        x = x.permute(0, 2, 1)
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)  # [B,1] logits


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def evaluate_with_sweep(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluate on a validation loader by sweeping thresholds in [0.05, 0.95]
    and picking the best F1.
    """
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


# -------------------------
# Helper: compute class imbalance for pos_weight
# -------------------------
def compute_pos_weight(dataset: Dataset) -> torch.Tensor | None:
    """
    Scan the dataset once to estimate positive / negative counts.
    Returns a tensor pos_weight = N_neg / N_pos for BCEWithLogitsLoss,
    or None if the dataset has no positives.
    """
    pos = 0
    total = 0
    for i in range(len(dataset)):
        _, y = dataset[i]
        y_val = float(y.item())
        total += 1
        if y_val >= 0.5:
            pos += 1

    if total == 0 or pos == 0:
        print("[warn] could not estimate pos_weight (no positives or empty dataset)")
        return None

    neg = total - pos
    if neg <= 0:
        print("[warn] dataset has no negatives; pos_weight not used")
        return None

    w = neg / pos
    print(f"[info] class balance: total={total}, pos={pos}, neg={neg}, pos_weight={w:.2f}")
    return torch.tensor([w], dtype=torch.float32)


# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train an enhanced TCN on windowed pose sequences.")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir",   required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=33724876)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience in epochs (on val F1).")
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

    # Model
    model = TCN(in_ch=C).to(device)

    # Loss with optional pos_weight for class imbalance
    pos_weight = compute_pos_weight(train_ds)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")
    epochs_no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train TCN ep{ep}", leave=False)
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

        # Validation sweep
        stats = evaluate_with_sweep(model, val_loader, device)
        note  = stats.get("note", "")
        print(f"val: P={stats['P']:.3f} R={stats['R']:.3f} F1={stats['F1']:.3f} @thr={stats['thr']:.2f} {note}")

        # Track best F1 and early stopping
        if stats["F1"] > best_f1 + 1e-6:
            best_f1 = stats["F1"]
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "in_ch": C, "best_thr": stats["thr"]},
                best_path,
            )
            print(f"[save] {best_path} (F1={best_f1:.3f} @thr={stats['thr']:.2f})")
        else:
            epochs_no_improve += 1
            print(f"[info] no improvement in F1 for {epochs_no_improve} epoch(s)")

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[early stop] patience={args.patience} reached at epoch {ep}")
            break

    print(f"[done] best F1={best_f1:.3f}  ckpt={best_path}")


if __name__ == "__main__":
    main()
