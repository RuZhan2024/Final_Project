#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse, numpy as np, yaml
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

# ---------- dataset utils ----------

def _label_from_npz(d) -> float | None:
    # Prefer numeric 'y'
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

    # Fallback to label-like fields
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
    return None

class WindowNPZ(Dataset):
    def __init__(self, root: str, skip_unlabeled=True):
        files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .npz under {root}")
        kept = []
        for p in files:
            try:
                with np.load(p, allow_pickle=False) as d:
                    y = _label_from_npz(d)
                if skip_unlabeled and y is None:
                    continue
                kept.append(p)
            except Exception:
                pass
        if not kept:
            raise FileNotFoundError(f"All windows under {root} are unlabeled or unreadable.")
        self.files = kept

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx], allow_pickle=False) as d:
            xy  = d["xy"].astype(np.float32)   # [W,33,2]
            conf= d["conf"].astype(np.float32) # [W,33]
            xy  = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
            x   = (xy * conf[..., None]).reshape(xy.shape[0], -1)  # [W,66]
            y   = _label_from_npz(d)
            if y is None: y = 0.0
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32)

# ---------- model ----------

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
        # x: [B, T, C] -> [B, C, T]
        x = x.permute(0,2,1)
        return self.head(self.net(x).squeeze(-1))

# ---------- evaluation & op pick ----------

@torch.no_grad()
def collect_probs_and_labels(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze(-1)
        probs  = torch.sigmoid(logits)
        ys.append(yb.cpu().numpy())
        ps.append(probs.cpu().numpy())
    y = np.concatenate(ys, axis=0).ravel().astype(int)
    p = np.concatenate(ps, axis=0).ravel()
    return p, y

def sweep_metrics(probs, y, sweep=401):
    thrs = np.linspace(0.01, 0.99, sweep)
    stats = []
    for t in thrs:
        yhat = (probs >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
        # FPR = FP / (FP+TN)
        tn = np.sum((y==0)&(yhat==0))
        fp = np.sum((y==0)&(yhat==1))
        fpr = fp / (fp + tn) if (fp+tn)>0 else 0.0
        stats.append((t, pr, rc, f1, fpr))
    return np.array(stats)  # [N,5]

def pick_ops(stats, recall_floor_op1=0.95, recall_floor_op3=0.90):
    # stats columns: thr, P, R, F1, FPR
    thr, P, R, F1, FPR = [stats[:,i] for i in range(5)]

    # OP2: max F1
    i2 = int(np.argmax(F1))
    op2 = dict(thr=float(thr[i2]), f1=float(F1[i2]), precision=float(P[i2]), recall=float(R[i2]))

    # OP1: highest recall meeting floor (tie-breaker: highest precision, then lowest thr)
    mask1 = R >= recall_floor_op1
    if mask1.any():
        idxs = np.where(mask1)[0]
        # sort by (-recall, -precision, thr)
        idxs = sorted(idxs, key=lambda i:(-R[i], -P[i], thr[i]))
        i1 = idxs[0]
    else:
        # fall back to max recall achievable
        i1 = int(np.argmax(R))
    op1 = dict(thr=float(thr[i1]), recall=float(R[i1]), precision=float(P[i1]))

    # OP3: min FPR subject to recall floor (tie-breaker: highest precision)
    mask3 = R >= recall_floor_op3
    if mask3.any():
        idxs = np.where(mask3)[0]
        i3 = int(idxs[np.argmin(FPR[idxs])])
    else:
        # fall back: pick global min FPR (will likely be high thr)
        i3 = int(np.argmin(FPR))
    op3 = dict(thr=float(thr[i3]), fpr=float(FPR[i3]), recall_at_thr=float(R[i3]), precision=float(P[i3]))

    return op1, op2, op3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recall_floor_op1", type=float, default=0.95)
    ap.add_argument("--recall_floor_op3", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    # Data (skip unlabeled)
    val_ds = WindowNPZ(args.val_dir, skip_unlabeled=True)
    x0,_ = val_ds[0]
    T, C = x0.shape

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # Model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_ch = ckpt.get("in_ch", C)
    model = TCN(in_ch=in_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    probs, y = collect_probs_and_labels(model, val_loader, device)

    if y.sum()==0:
        # safety: no positives in val ⇒ choose conservative defaults
        op1 = dict(thr=0.5, recall=0.0, precision=0.0, note="no positives in val")
        op2 = dict(thr=0.5, f1=0.0, precision=0.0, recall=0.0, note="no positives in val")
        op3 = dict(thr=0.9, fpr=0.0, recall_at_thr=0.0, precision=0.0, note="no positives in val")
    else:
        stats = sweep_metrics(probs, y)
        op1, op2, op3 = pick_ops(stats, args.recall_floor_op1, args.recall_floor_op3)

    ops = {"OP1_high_recall": op1, "OP2_balanced": op2, "OP3_low_alarm": op3}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(ops, f, sort_keys=False)

    print("[ops]", ops)

if __name__ == "__main__":
    main()
