#!/usr/bin/env python3
"""
Score unlabeled windows and estimate alert rate per 24h.

Inputs:
  --windows_dir   e.g. data/processed/le2i/windows_W48_S12/test_unlabeled
  --ckpt          trained model checkpoint (from train_tcn.py, stores 'model' and 'in_ch')
  --thr           decision threshold (e.g., from OP-3 picked on labeled val)
  --fps           frames per second for LE2i (25 or 30; if unknown, try 25)
  --cooldown_sec  gap to merge consecutive alert windows into one 'event' (default 3s)

Output:
  - prints per-scene and overall: alerts, hours, alerts/24h
  - writes CSV with per-window scores for optional review
"""

import os, re, glob, csv, argparse, pathlib, collections
import numpy as np
import torch, torch.nn as nn

# ------- Must match your TCN definition used in training -------
class TCN(nn.Module):
    def __init__(self, in_ch, hid=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=5, padding=2), nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(hid, hid, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(hid, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,C,T]
        h = self.net(x).squeeze(-1)
        return self.head(h)
# ---------------------------------------------------------------

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    in_ch = ck.get('in_ch')
    m = TCN(in_ch=in_ch)
    m.load_state_dict(ck['model'], strict=True)
    m.to(device).eval()
    return m, in_ch

def read_window(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    xy   = np.nan_to_num(z["xy"].astype(np.float32))
    conf = np.nan_to_num(z["conf"].astype(np.float32))
    # feature: [T, J*2]
    x = (xy * conf[..., None]).reshape(xy.shape[0], -1)
    # metadata
    W = xy.shape[0]
    stem = z["stem"].item() if "stem" in z.files else pathlib.Path(npz_path).stem.split("_t")[0]
    start = int(z["start"]) if "start" in z.files else infer_start_from_name(npz_path)
    return x, stem, start, W

def infer_start_from_name(p):
    m = re.search(r"_t(\d+)\.npz$", os.path.basename(p))
    return int(m.group(1)) if m else 0

def group_alerts(starts, W, stride, fps, cooldown_sec):
    """Merge consecutive alert windows into events with a time gap > cooldown."""
    if not starts: return 0
    starts = sorted(starts)
    gap_frames = int(cooldown_sec * fps)
    events = 1
    last_end = starts[0] + W - 1
    for s in starts[1:]:
        # If new alert begins after a cooldown gap from last_end, new event
        if s - last_end > gap_frames:
            events += 1
        last_end = s + W - 1
    return events

def estimate_video_duration_from_windows(starts, W, stride, fps):
    """Approximate total seconds of the video from the max covered frame."""
    if not starts: return 0.0
    T_est = max(starts) + W  # frames
    return T_est / float(fps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=12, help="stride used when windowing")
    ap.add_argument("--cooldown_sec", type=float, default=3.0)
    ap.add_argument("--csv_out", default="outputs/reports/unlabeled_scores.csv")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, in_ch = load_model(args.ckpt, device)

    files = sorted(glob.glob(os.path.join(args.windows_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No windows in {args.windows_dir}")

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    per_scene_starts = collections.defaultdict(list)
    per_scene_W = {}
    per_scene_secs = collections.defaultdict(float)
    total_rows = 0

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "stem", "start", "W", "p_fall", "alert"])
        # score windows
        for i, p in enumerate(files):
            x, stem, start, W = read_window(p)
            X = torch.from_numpy(x).unsqueeze(0).to(device)      # [1, T, C]
            logits = model(X)                                     # [1,1]
            pfall = torch.sigmoid(logits).item()
            alert = int(pfall >= args.thr)
            w.writerow([os.path.basename(p), stem, start, W, f"{pfall:.6f}", alert])
            total_rows += 1
            if alert:
                per_scene_starts[stem].append(start)
                per_scene_W[stem] = W

        # estimate per-scene duration from windows we saw
        by_stem_starts = collections.defaultdict(list)
        for p in files:
            stem = pathlib.Path(p).stem.split("_t")[0]
            s = infer_start_from_name(p)
            by_stem_starts[stem].append(s)
            # grab W from earlier dict if present; if not, read once
            if stem not in per_scene_W:
                # read one file of this stem to get W
                x, _, _, Wtmp = read_window(p)
                per_scene_W[stem] = Wtmp

        for stem, starts in by_stem_starts.items():
            secs = estimate_video_duration_from_windows(starts, per_scene_W[stem], args.stride, args.fps)
            per_scene_secs[stem] = secs

    # Summaries
    total_hours = sum(per_scene_secs.values()) / 3600.0
    total_events = 0
    per_scene_events = {}
    for stem, starts in per_scene_starts.items():
        events = group_alerts(starts, per_scene_W[stem], args.stride, args.fps, args.cooldown_sec)
        per_scene_events[stem] = events
        total_events += events

    print("\n=== Unlabeled Alert Summary ===")
    print(f"Windows scored: {total_rows}")
    print(f"Coverage: {total_hours:.2f} hours")
    print(f"Alerts (grouped by {args.cooldown_sec}s cooldown): {total_events}")
    if total_hours > 0:
        print(f"Alerts per 24h: {total_events / total_hours * 24:.3f}")

    # per-scene table
    print("\nPer-scene:")
    for stem in sorted(per_scene_secs):
        hours = per_scene_secs[stem] / 3600.0
        ev = per_scene_events.get(stem, 0)
        rate = (ev / hours * 24) if hours > 0 else 0.0
        print(f"  {stem:30s}  hours={hours:6.2f}  alerts={ev:4d}  alerts/24h={rate:6.3f}")

    print(f"\n[Wrote per-window scores] {args.csv_out}")

if __name__ == "__main__":
    main()
