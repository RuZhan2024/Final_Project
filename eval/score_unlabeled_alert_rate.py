
"""
Score unlabeled windows and estimate alert rate per 24h.

Supports:
  - TCN windows: x [T,66] (flattened 33x2, gated by conf)
  - GCN windows: x [T,33,4] (pelvis-centred xy + velocity, gated by conf)

Inputs:
  --windows_dir     e.g. data/processed/le2i/windows_W48_S12/test_unlabeled
  --ckpt            trained model checkpoint
  --arch            tcn | gcn
  --thr             decision threshold
  --fps             fallback fps if window NPZ lacks fps
  --fps_default     (GCN) fallback fps if window NPZ lacks fps
  --cooldown_sec    merge consecutive alert windows into one 'event' (default 3s)

Output:
  - prints per-scene and overall: alerts, hours, alerts/24h
  - writes CSV with per-window scores for optional review

Compatibility:
  - Works with window filenames like:
      (new)  <video_id>__w000000_000047.npz
      (old)  <stem>_t123.npz
  - Works with NPZ keys:
      (preferred) video_id + start + fps
      (legacy)    stem + start + fps
"""

from __future__ import annotations

import os
import re
import glob
import csv
import argparse
import pathlib
import collections
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Shared helpers
# ----------------------------
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


def infer_start_from_name(p: str) -> int:
    """
    Support both naming styles:
      - new:  <video_id>__w000123_000170.npz  -> start=123
      - old:  <stem>_t123.npz                -> start=123
    """
    base = os.path.basename(p)

    m = re.search(r"__w(\d+)_\d+\.npz$", base)
    if m:
        return int(m.group(1))

    m = re.search(r"_t(\d+)\.npz$", base)
    if m:
        return int(m.group(1))

    return 0


def infer_id_from_name(p: str) -> str:
    """
    Infer the sequence id (stem/video_id) from the filename if NPZ doesn't have it.
    """
    name = pathlib.Path(p).stem  # without .npz
    if "__w" in name:
        return name.split("__w", 1)[0]
    if "_t" in name:
        return name.split("_t", 1)[0]
    return name


@torch.no_grad()
def p_fall_from_logits(logits: torch.Tensor, prob_mode: str) -> float:
    """
    prob_mode:
      - "sigmoid": logits are 1-logit (shape [B] or [B,1])
      - "softmax": logits are 2-logit (shape [B,2])
    """
    if prob_mode == "sigmoid":
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return float(torch.sigmoid(logits).item())
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


def group_alerts(starts: List[int], W: int, fps: float, cooldown_sec: float) -> int:
    """Merge consecutive alert windows into events with a time gap > cooldown."""
    if not starts:
        return 0
    starts = sorted(starts)
    gap_frames = int(cooldown_sec * fps)
    events = 1
    last_end = starts[0] + W - 1
    for s in starts[1:]:
        if s - last_end > gap_frames:
            events += 1
        last_end = s + W - 1
    return events


def estimate_video_duration_from_windows(starts: List[int], W: int, fps: float) -> float:
    """Approximate total seconds of the video from the max covered frame."""
    if not starts:
        return 0.0
    T_est = max(starts) + W
    return float(T_est) / float(fps)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# TCN models (same idea as metrics.py)
# ----------------------------
class SimpleTCN(nn.Module):
    def __init__(self, in_ch=66, hidden=128, dropout=0.2, out_dim=1):
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
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        kernel_in: int = 5,
        kernel_block: int = 3,
        dilations=None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        pad_in = (kernel_in - 1) // 2
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=kernel_in, padding=pad_in),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResTCNBlock(hidden, kernel_size=kernel_block, dilation=int(d), p=dropout) for d in dilations]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,C,T]
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        h = x.mean(dim=-1)
        return self.head(h)


def _infer_num_blocks(sd: dict) -> int:
    idxs = []
    for k in sd.keys():
        if isinstance(k, str) and k.startswith("blocks.") and ".conv.weight" in k:
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0


def _infer_dilations(ckpt: dict, n_blocks: int):
    d = ckpt.get("dilations", None)
    if isinstance(d, (list, tuple)) and len(d) == n_blocks:
        return [int(x) for x in d]
    return [2 ** i for i in range(n_blocks)] if n_blocks > 0 else [1, 2, 4]


def build_tcn_from_ckpt(ckpt: dict):
    sd = _get_state_dict(ckpt)

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

    if "net.0.weight" in sd:
        in_ch = int(sd["net.0.weight"].shape[1])
        hidden = int(sd["net.0.weight"].shape[0])
    else:
        in_ch = int(ckpt.get("in_ch", 66))
        hidden = int(ckpt.get("hidden", 128))

    out_dim = int(sd["head.weight"].shape[0]) if "head.weight" in sd else 1
    model = SimpleTCN(in_ch=in_ch, hidden=hidden, out_dim=out_dim, dropout=float(ckpt.get("dropout", 0.2)))
    prob_mode = "sigmoid" if out_dim == 1 else "softmax"
    return model, prob_mode


# ----------------------------
# GCN models (two styles)
# ----------------------------
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


class BNGraphBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.3):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # x: [B,T,V,C]
        x = torch.einsum("vw,btwc->btvc", A_hat, x)
        B, T, V, _ = x.shape
        x = self.lin(x)  # [B,T,V,Cout]
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
        x = x.mean(dim=2)      # [B,T,gcn_out]
        x = x.permute(0, 2, 1) # [B,gcn_out,T]
        h = self.temporal(x).squeeze(-1)
        return self.head(h)


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


def build_gcn_from_ckpt(ckpt: dict):
    sd = _get_state_dict(ckpt)

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
        prob_mode = "sigmoid" if out_dim == 1 else "softmax"
        return model, prob_mode

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


# ----------------------------
# Window IO
# ----------------------------
def _npz_get_str(z, key: str) -> str | None:
    if key not in z.files:
        return None
    v = z[key]
    try:
        return str(v.item())
    except Exception:
        try:
            return str(np.array(v).reshape(-1)[0])
        except Exception:
            return None


def read_window_tcn(npz_path: str, fps_fallback: float):
    with np.load(npz_path, allow_pickle=False) as z:
        xy = np.nan_to_num(z["xy"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(z["conf"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        fps = float(z["fps"]) if "fps" in z.files else float(fps_fallback)

        # Prefer stable id from NPZ
        seq_id = _npz_get_str(z, "video_id") or _npz_get_str(z, "stem") or infer_id_from_name(npz_path)
        start = int(z["start"]) if "start" in z.files else infer_start_from_name(npz_path)

    x = (xy * conf[..., None]).reshape(xy.shape[0], -1)  # [T,66]
    W = int(xy.shape[0])
    return x, seq_id, start, W, fps


def read_window_gcn(npz_path: str, fps_default: float):
    with np.load(npz_path, allow_pickle=False) as z:
        xy = np.nan_to_num(z["xy"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(z["conf"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        fps = float(z["fps"]) if "fps" in z.files else float(fps_default)

        seq_id = _npz_get_str(z, "video_id") or _npz_get_str(z, "stem") or infer_id_from_name(npz_path)
        start = int(z["start"]) if "start" in z.files else infer_start_from_name(npz_path)

    x = xy * conf[..., None]          # [T,33,2]
    pelvis = x[:, 23:24, :]           # [T,1,2]
    x_rel = x - pelvis                # [T,33,2]

    vel = np.zeros_like(x_rel)
    vel[1:] = (x_rel[1:] - x_rel[:-1]) * fps

    feats = np.concatenate([x_rel, vel], axis=-1)  # [T,33,4]
    W = int(xy.shape[0])
    return feats, seq_id, start, W, fps


# ----------------------------
# Model loader
# ----------------------------
def load_model(ckpt_path: str, arch: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ck)

    if arch == "tcn":
        model, prob_mode = build_tcn_from_ckpt(ck)
    elif arch == "gcn":
        model, prob_mode = build_gcn_from_ckpt(ck)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, prob_mode


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, required=True)

    ap.add_argument("--arch", choices=["tcn", "gcn"], default="tcn")

    ap.add_argument("--fps", type=float, default=25.0, help="fallback fps if window npz lacks fps (TCN path)")
    ap.add_argument("--fps_default", type=float, default=30.0, help="fallback fps if window npz lacks fps (GCN path)")

    ap.add_argument("--stride", type=int, default=12, help="kept for CLI compatibility (not used for duration here)")
    ap.add_argument("--cooldown_sec", type=float, default=3.0)
    ap.add_argument("--csv_out", default="outputs/reports/unlabeled_scores.csv")
    args = ap.parse_args()

    device = pick_device()
    model, prob_mode = load_model(args.ckpt, arch=args.arch, device=device)

    files = sorted(glob.glob(os.path.join(args.windows_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No windows in {args.windows_dir}")

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)

    # Group by seq_id (scene)
    by_id_all_starts = collections.defaultdict(list)
    by_id_alert_starts = collections.defaultdict(list)
    by_id_W: Dict[str, int] = {}
    by_id_fps: Dict[str, float] = {}

    total_rows = 0

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "video_id", "start", "W", "fps", "p_fall", "alert"])

        for p in files:
            if args.arch == "tcn":
                x, vid, start, W, fps = read_window_tcn(p, fps_fallback=args.fps)
                X = torch.from_numpy(x).unsqueeze(0).to(device)       # [1,T,66]
            else:
                x, vid, start, W, fps = read_window_gcn(p, fps_default=args.fps_default)
                X = torch.from_numpy(x).unsqueeze(0).to(device)       # [1,T,33,4]

            logits = model(X)
            pfall = p_fall_from_logits(logits, prob_mode=prob_mode)
            alert = int(pfall >= args.thr)

            w.writerow([os.path.basename(p), vid, start, W, f"{fps:.3f}", f"{pfall:.6f}", alert])
            total_rows += 1

            by_id_all_starts[vid].append(start)
            by_id_W.setdefault(vid, W)
            by_id_fps.setdefault(vid, fps)
            if alert:
                by_id_alert_starts[vid].append(start)

    total_secs = 0.0
    total_events = 0

    per_id_secs = {}
    per_id_events = {}

    for vid, starts in by_id_all_starts.items():
        W = by_id_W[vid]
        fps = by_id_fps.get(vid, args.fps if args.arch == "tcn" else args.fps_default)

        secs = estimate_video_duration_from_windows(starts, W, fps)
        per_id_secs[vid] = secs
        total_secs += secs

        events = group_alerts(by_id_alert_starts.get(vid, []), W, fps, args.cooldown_sec)
        per_id_events[vid] = events
        total_events += events

    total_hours = total_secs / 3600.0

    print("\n=== Unlabeled Alert Summary ===")
    print(f"Arch: {args.arch}  (prob_mode={prob_mode})")
    print(f"Windows scored: {total_rows}")
    print(f"Coverage: {total_hours:.2f} hours")
    print(f"Alerts (grouped by {args.cooldown_sec}s cooldown): {total_events}")
    if total_hours > 0:
        print(f"Alerts per 24h: {total_events / total_hours * 24:.3f}")

    print("\nPer-scene:")
    for vid in sorted(per_id_secs):
        hours = per_id_secs[vid] / 3600.0
        ev = per_id_events.get(vid, 0)
        rate = (ev / hours * 24) if hours > 0 else 0.0
        fps = by_id_fps.get(vid, args.fps if args.arch == "tcn" else args.fps_default)
        print(f"  {vid:30s}  fps={fps:5.1f}  hours={hours:6.2f}  alerts={ev:4d}  alerts/24h={rate:6.3f}")

    print(f"\n[Wrote per-window scores] {args.csv_out}")


if __name__ == "__main__":
    main()
