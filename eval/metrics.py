#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse, json, re
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import yaml

# ------------------ label utils ------------------

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

# ------------------ dataset ------------------

class WindowNPZ(Dataset):
    """
    Returns (x, y, meta_dict):
      x: [W, C] float32 (C=66)
      y: [1] float32 (0/1), unlabeled skipped by default
      meta: {'stem': str, 'start': int, 'end': int, 'path': str}
            start/end use -1 when unknown (no None).
    """
    def __init__(self, root: str, skip_unlabeled: bool = True):
        files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .npz under {root}")
        self.items = []
        skipped = 0
        for p in files:
            try:
                with np.load(p, allow_pickle=False) as d:
                    y = _label_from_npz(d)
                if skip_unlabeled and y is None:
                    skipped += 1
                    continue
                self.items.append(p)
            except Exception:
                skipped += 1
        if not self.items:
            raise FileNotFoundError(f"All windows under {root} are unlabeled or unreadable.")
        if skipped:
            print(f"[metrics] skipped {skipped} unlabeled/unreadable windows under {root}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]
        with np.load(path, allow_pickle=False) as d:
            xy   = d["xy"].astype(np.float32)   # [W,33,2]
            conf = d["conf"].astype(np.float32) # [W,33]
            xy   = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
            x    = (xy * conf[..., None]).reshape(xy.shape[0], -1)  # [W,66]
            y    = _label_from_npz(d)
            if y is None: y = 0.0

            # --- sanitized meta (no None) ---
            start = int(d["start"]) if "start" in d.files else -1
            end   = int(d["end"])   if "end"   in d.files else -1
            stem  = None
            if "video_id" in d.files:
                v = d["video_id"]
                if isinstance(v, bytes): v = v.decode()
                elif isinstance(v, np.ndarray):
                    try: v = v.item()
                    except Exception: v = str(v)
                stem = str(v)
            if stem is None:
                base = os.path.basename(path)
                m = re.search(r"^(.*)__w(\d+)_([0-9]+)\.npz$", base)
                if m:
                    stem = m.group(1)
                    if start < 0: start = int(m.group(2))
                    if end   < 0: end   = int(m.group(3))
                else:
                    stem = os.path.splitext(base)[0]
            meta = {"stem": stem, "start": int(start), "end": int(end), "path": path}
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32), meta

# ------------------ model ------------------

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
    def forward(self, x):
        x = x.permute(0,2,1)
        return self.head(self.net(x).squeeze(-1))

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --------------- collate that keeps metas as a list ---------------

def collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)  # tuples
    xs = torch.stack(xs, 0)      # [B, T, C]
    ys = torch.stack(ys, 0)      # [B, 1]
    metas = list(metas)          # leave metas un-collated
    return xs, ys, metas

# ------------------ evaluation ------------------

@torch.no_grad()
def predict_probs(ckpt_path: str, ds: Dataset, batch=256):
    x0, _, _ = ds[0]
    T, C = x0.shape
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_ch = ckpt.get("in_ch", C)
    model = TCN(in_ch=in_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=collate_keep_meta)
    probs, ys, metas = [], [], []
    for xb, yb, mb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)
        pb = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(pb)
        ys.append(yb.numpy())
        metas.extend(mb)
    p = np.concatenate(probs, axis=0).ravel()
    y = np.concatenate(ys, axis=0).ravel().astype(int)
    return p, y, metas

def prf_at_thr(p, y, thr):
    yhat = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return float(pr), float(rc), float(f1)

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ops", required=True, help="configs/ops.yaml")
    ap.add_argument("--fps", type=float, default=30.0)  # kept for future FA/24h calc
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    # data + predictions
    ds = WindowNPZ(args.eval_dir, skip_unlabeled=True)
    p, y, metas = predict_probs(args.ckpt, ds)

    # load OPs
    ops = yaml.safe_load(open(args.ops, "r"))
    thr1 = float(ops["OP1_high_recall"]["thr"])
    thr2 = float(ops["OP2_balanced"]["thr"])
    thr3 = float(ops["OP3_low_alarm"]["thr"])

    # window-level metrics
    p1, r1, f1 = prf_at_thr(p, y, thr1)
    p2, r2, f2 = prf_at_thr(p, y, thr2)
    p3, r3, f3 = prf_at_thr(p, y, thr3)

    have_meta = any((m.get("start", -1) >= 0) and (m.get("end", -1) >= 0) for m in metas)
    report = {
        "dataset": os.path.basename(args.eval_dir.rstrip("/")),
        "n_windows": int(len(y)),
        "pos_windows": int(y.sum()),
        "ops": {
            "OP1_high_recall": {"thr": thr1, "precision": p1, "recall": r1, "f1": f1},
            "OP2_balanced":    {"thr": thr2, "precision": p2, "recall": r2, "f1": f2},
            "OP3_low_alarm":   {"thr": thr3, "precision": p3, "recall": r3, "f1": f3},
        }
    }
    if not have_meta:
        report["note"] = "FA/24h not computed (windows missing start/end metadata)."

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("[report]", args.report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
