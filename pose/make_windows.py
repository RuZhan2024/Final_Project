#!/usr/bin/env python3
# Make fixed-length windows; label a window positive if it overlaps any fall span.

import os, argparse, glob, json, pathlib
import numpy as np

def load_json(path, default=None):
    if not path:
        return default
    with open(path, "r") as f:
        return json.load(f)

def list_npzs(npz_dir):
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))

def load_list(path):
    if not path:
        return None
    with open(path, "r") as f:
        return set(x.strip() for x in f if x.strip())

def decide_split(stem, train_set, val_set, test_set):
    # No split files provided → put everything under "unsplit"
    if train_set is None and val_set is None and test_set is None:
        return "unsplit"
    if train_set and stem in train_set:
        return "train"
    if val_set and stem in val_set:
        return "val"
    if test_set and stem in test_set:
        return "test"
    return None  # skip if lists provided but stem not listed

def iter_windows(xy, conf, W, stride):
    T = xy.shape[0]
    if T < W:
        return
    for start in range(0, T - W + 1, stride):
        yield start, xy[start:start+W], conf[start:start+W]

def overlaps(window_start, window_end, spans):
    for s, e in spans:
        # intersection non-empty
        if not (window_end < s or e < window_start):
            return True
    return False

def normalize_spans(spans_obj):
    """
    Ensure spans dict is stem -> list of [int start, int end].
    Accepts ints/strings; ignores malformed entries.
    """
    out = {}
    if not isinstance(spans_obj, dict):
        return out
    for stem, lst in spans_obj.items():
        good = []
        if isinstance(lst, (list, tuple)):
            for item in lst:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                try:
                    s = int(item[0]); e = int(item[1])
                except Exception:
                    continue
                if e >= s:
                    good.append([s, e])
        if good:
            out[stem] = good
    return out

def main():
    ap = argparse.ArgumentParser(description="Create sliding windows with span-aware labels.")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_json", required=True, help="video-level labels (adl/fall)")
    ap.add_argument("--spans_json", default=None, help="optional: stem -> [[start,end],...]")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--train_list", type=str, default=None)
    ap.add_argument("--val_list",   type=str, default=None)
    ap.add_argument("--test_list",  type=str, default=None)
    args = ap.parse_args()

    labels_raw = load_json(args.labels_json, {})
    # labels: stem -> "adl"/"fall"
    labels = {k: str(v).lower() for k, v in (labels_raw or {}).items()}
    spans_raw = load_json(args.spans_json, {}) if args.spans_json else {}
    spans = normalize_spans(spans_raw)

    train_set = load_list(args.train_list)
    val_set   = load_list(args.val_list)
    test_set  = load_list(args.test_list)

    files = list_npzs(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No NPZs under {args.npz_dir}")

    # Prepare output subfolders
    subdirs = ["train","val","test"] if (train_set or val_set or test_set) else ["unsplit"]
    for sd in subdirs:
        os.makedirs(os.path.join(args.out_dir, sd), exist_ok=True)

    total = {sd: 0 for sd in subdirs}
    saved = 0
    missing_in_labels = 0

    for seq_npz in files:
        stem = pathlib.Path(seq_npz).stem
        split = decide_split(stem, train_set, val_set, test_set)
        if split is None:
            # Not listed in provided splits → skip
            continue

        with np.load(seq_npz, allow_pickle=False) as d:
            xy   = np.nan_to_num(d["xy"].astype(np.float32),   nan=0.0, posinf=0.0, neginf=0.0)
            conf = np.nan_to_num(d["conf"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # default video-level label: 'fall' -> 1 else 0
        video_label = 1 if labels.get(stem, "adl") == "fall" else 0
        seq_spans = spans.get(stem, None)  # None or list[[s,e],...]

        wrote_any = False
        for start, w_xy, w_conf in iter_windows(xy, conf, args.W, args.stride):
            end = start + args.W - 1
            if seq_spans:
                y = 1 if overlaps(start, end, seq_spans) else 0
            else:
                y = video_label

            out_name = f"{stem}__w{start:06d}_{end:06d}.npz"
            out_path = os.path.join(args.out_dir, split, out_name)
            np.savez_compressed(
                out_path,
                xy=w_xy,                  # [W,33,2]
                conf=w_conf,              # [W,33]
                y=int(y),                 # 0/1
                start=int(start),         # window start (frame idx)
                end=int(end),             # window end (frame idx)
                video_id=str(stem)        # parent sequence id
            )
            total[split] += 1
            saved += 1
            wrote_any = True

        if stem not in labels:
            missing_in_labels += 1

    print(f"[done] saved {saved} windows → {args.out_dir}")
    for k, v in total.items():
        print(f"  {k}: {v}")
    if missing_in_labels:
        print(f"[warn] {missing_in_labels} stems not in labels_json (defaulted to ADL)")

if __name__ == "__main__":
    main()
