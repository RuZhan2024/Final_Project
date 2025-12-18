
"""
make_windows.py

Create fixed-length windows from pose NPZ sequences and assign window labels.

Inputs
------
1) Sequence NPZ files under --npz_dir (recursive). Each sequence NPZ should contain:
   - xy   : [T, J, 2] float (2D joints)
   - conf : [T, J]    float (joint confidence)
   - optionally fps   : scalar float/int

2) Video-level labels JSON (--labels_json):
   { "<stem>": "fall" | "adl" | 1 | 0 | ... }

3) Optional fall spans JSON (--spans_json):
   { "<stem>": [[start, end], ...] }

Span semantics
--------------
By default we interpret each span as inclusive endpoints: [start, end] (both inclusive).
If --spans_end_exclusive is set, we interpret spans as half-open intervals: [start, end).

Window semantics
----------------
We store window metadata as:
- start : inclusive
- end   : inclusive

Overlap rule
------------
If spans exist for a stem, we label a window positive (y=1) iff the overlap length
(in frames) between window and any fall span is >= required_overlap_frames.

required_overlap_frames is:
- if --min_overlap_frames is set: that value
- else if --min_overlap_frac is set: ceil(min_overlap_frac * W)
- else: 1  (any overlap)

If no spans exist for a stem, we fall back to the video-level label (from labels_json).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# Utilities
# -------------------------

def load_json(path: Optional[str], default=None):
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_npzs(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))


def load_list(path: Optional[str]) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.add(s)
    return out


def stem_from_path(p: str) -> str:
    """Use filename stem as the sequence identifier."""
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return stem


def normalise_label_to_y(v) -> int:
    """
    Convert a label value to y in {0,1}.

    Accepts strings like: "fall"/"adl", "1"/"0", "true"/"false".
    """
    if v is None:
        return 0
    if isinstance(v, (int, np.integer)):
        return 1 if int(v) != 0 else 0
    if isinstance(v, (float, np.floating)):
        return 1 if float(v) >= 0.5 else 0

    s = str(v).strip().lower()
    if s in ("fall", "positive", "pos", "yes", "true", "t", "1"):
        return 1
    if s in ("adl", "nonfall", "no", "false", "f", "0"):
        return 0

    # Fallback: try numeric parsing
    try:
        x = float(s)
        return 1 if x >= 0.5 else 0
    except Exception:
        return 0


def safe_scalar_fps(npz_obj: np.lib.npyio.NpzFile, fps_default: float) -> float:
    """
    Extract FPS from an NPZ if present. Otherwise return fps_default.
    """
    if "fps" in npz_obj.files:
        fps_val = npz_obj["fps"]
        try:
            return float(np.array(fps_val).reshape(-1)[0])
        except Exception:
            return float(fps_default)
    return float(fps_default)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -------------------------
# Spans and overlap
# -------------------------

def normalise_spans(raw: dict) -> Dict[str, List[Tuple[int, int]]]:
    """
    Ensure spans are {stem: [(start,end), ...]} with ints.
    Input can be lists of lists, tuples, etc.
    """
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not raw:
        return out
    for k, v in raw.items():
        if not v:
            continue
        spans: List[Tuple[int, int]] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    a = int(item[0])
                    b = int(item[1])
                    spans.append((a, b))
                except Exception:
                    continue
        if spans:
            out[str(k)] = spans
    return out


def overlap_len_inclusive(a0: int, a1: int, b0: int, b1: int) -> int:
    """Length of overlap between [a0,a1] and [b0,b1] inclusive."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def overlap_len_exclusive(a0: int, a1_excl: int, b0: int, b1_excl: int) -> int:
    """Length of overlap between [a0,a1) and [b0,b1) (exclusive end)."""
    lo = max(a0, b0)
    hi = min(a1_excl, b1_excl)
    return max(0, hi - lo)


def window_is_positive(
    w_start: int,
    w_end_inclusive: int,
    spans: Sequence[Tuple[int, int]],
    spans_end_exclusive: bool,
    required_overlap_frames: int,
) -> bool:
    """
    Determine if a window overlaps any fall span by at least required_overlap_frames.
    """
    if required_overlap_frames <= 0:
        required_overlap_frames = 1

    if spans_end_exclusive:
        # Convert window [start,end] -> [start, end+1)
        w0 = w_start
        w1 = w_end_inclusive + 1
        for s0, s1 in spans:
            # spans are [s0, s1) already
            if overlap_len_exclusive(w0, w1, s0, s1) >= required_overlap_frames:
                return True
        return False

    # Inclusive spans
    for s0, s1 in spans:
        if overlap_len_inclusive(w_start, w_end_inclusive, s0, s1) >= required_overlap_frames:
            return True
    return False


def compute_required_overlap_frames(
    W: int,
    min_overlap_frames: Optional[int],
    min_overlap_frac: Optional[float],
) -> int:
    if min_overlap_frames is not None:
        return max(1, int(min_overlap_frames))

    if min_overlap_frac is not None:
        frac = float(min_overlap_frac)
        if frac <= 0:
            return 1
        return max(1, int(math.ceil(frac * W)))

    return 1


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Directory containing sequence NPZs (recursive).")
    ap.add_argument("--labels_json", required=True, help="JSON mapping stem -> label (fall/adl).")
    ap.add_argument("--spans_json", default=None, help="Optional JSON mapping stem -> fall spans.")
    ap.add_argument("--out_dir", required=True, help="Output directory for window NPZs.")
    ap.add_argument("--W", type=int, required=True, help="Window length (frames).")
    ap.add_argument("--stride", type=int, required=True, help="Stride (frames).")
    ap.add_argument("--fps_default", type=float, default=30.0, help="Fallback FPS if sequence NPZ lacks fps.")
    ap.add_argument("--train_list", default=None, help="TXT list of stems for train split.")
    ap.add_argument("--val_list", default=None, help="TXT list of stems for val split.")
    ap.add_argument("--test_list", default=None, help="TXT list of stems for test split.")

    # Span controls (new)
    ap.add_argument(
        "--spans_end_exclusive",
        action="store_true",
        help="Interpret spans as [start,end) instead of inclusive [start,end].",
    )
    ap.add_argument(
        "--min_overlap_frac",
        type=float,
        default=None,
        help="Require at least this fraction of the window to overlap a fall span (e.g. 0.25).",
    )
    ap.add_argument(
        "--min_overlap_frames",
        type=int,
        default=None,
        help="Require at least this many frames of overlap with a fall span.",
    )

    args = ap.parse_args()

    if args.W <= 0 or args.stride <= 0:
        raise SystemExit("[ERR] W and stride must be positive integers.")

    labels_raw = load_json(args.labels_json, {})
    labels_y: Dict[str, int] = {str(k): normalise_label_to_y(v) for k, v in (labels_raw or {}).items()}

    spans_raw = load_json(args.spans_json, {}) if args.spans_json else {}
    spans = normalise_spans(spans_raw)

    train_set = load_list(args.train_list)
    val_set = load_list(args.val_list)
    test_set = load_list(args.test_list)

    files = list_npzs(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No NPZs found under: {args.npz_dir}")

    # Decide output subfolders
    use_splits = bool(train_set or val_set or test_set)
    subdirs = ["train", "val", "test"] if use_splits else ["unsplit"]
    for sd in subdirs:
        ensure_dir(os.path.join(args.out_dir, sd))

    required_overlap = compute_required_overlap_frames(args.W, args.min_overlap_frames, args.min_overlap_frac)

    saved = 0
    missing_in_labels = 0
    counts = {sd: 0 for sd in subdirs}

    for seq_path in files:
        stem = stem_from_path(seq_path)

        if use_splits:
            if stem in train_set:
                split = "train"
            elif stem in val_set:
                split = "val"
            elif stem in test_set:
                split = "test"
            else:
                # If you pass split lists, we skip unknown stems by design.
                continue
        else:
            split = "unsplit"

        with np.load(seq_path) as d:
            if "xy" not in d.files or "conf" not in d.files:
                # Not a pose sequence NPZ
                continue

            xy = d["xy"]
            conf = d["conf"]

            if xy.ndim != 3:
                raise SystemExit(f"[ERR] {seq_path}: expected xy [T,J,2], got shape {xy.shape}")
            if conf.ndim != 2:
                raise SystemExit(f"[ERR] {seq_path}: expected conf [T,J], got shape {conf.shape}")

            T = int(xy.shape[0])
            fps = safe_scalar_fps(d, args.fps_default)

        # Video-level label fallback (if spans missing)
        if stem in labels_y:
            video_y = int(labels_y[stem])
        else:
            video_y = 0
            missing_in_labels += 1

        seq_spans = spans.get(stem, [])

        # If sequence shorter than a single window, skip
        if T < args.W:
            continue

        # Create windows
        for start in range(0, T - args.W + 1, args.stride):
            end = start + args.W - 1  # inclusive

            if seq_spans:
                y = 1 if window_is_positive(
                    w_start=start,
                    w_end_inclusive=end,
                    spans=seq_spans,
                    spans_end_exclusive=args.spans_end_exclusive,
                    required_overlap_frames=required_overlap,
                ) else 0
            else:
                y = video_y

            out_name = f"{stem}__w{start:06d}_{end:06d}.npz"
            out_path = os.path.join(args.out_dir, split, out_name)

            # Save (keep same keys expected by downstream)
            np.savez_compressed(
                out_path,
                xy=np.nan_to_num(xy[start:end + 1]),
                conf=np.nan_to_num(conf[start:end + 1]),
                y=np.int64(y),
                start=np.int64(start),
                end=np.int64(end),
                video_id=stem,
                fps=np.float32(fps),
            )

            counts[split] += 1
            saved += 1

    print(f"[OK] saved {saved} windows → {args.out_dir}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if missing_in_labels:
        print(f"[warn] {missing_in_labels} stems not in labels_json (defaulted to ADL)")


if __name__ == "__main__":
    main()
