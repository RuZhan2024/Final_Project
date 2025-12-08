#!/usr/bin/env python3
"""
Extract 2D human pose from **image sequences** (e.g., URFD/CAUCAFall/MUVIM frame folders)
using MediaPipe Pose and save **one NPZ per sequence**.

Each output NPZ stores:
  - xy  : float32 [T, 33, 2]  → normalized landmark coords (x, y in [0,1])
  - conf: float32 [T, 33]     → landmark visibility/confidence proxy
  - y   : float32 scalar      → 1.0 for fall, 0.0 for adl, -1.0 for unlabeled

Notes
-----
- We group frames into a "sequence" using `sequence_id_depth` (how many parent dirs above
  the image filename to include in the sequence ID).
- For **URFD** we typically have paths like:
      data/raw/UR_Fall/adl/adl-01-cam0-rgb/adl-01-cam0-rgb-001.png
  with label in the directory name ("adl" / "fall").
- For **CAUCAFall** we infer labels by checking the last path component (sequence name)
  and looking for the substring "fall" (case-insensitive). Everything else is "adl".
- For **MUVIM ZED_RGB** we rely on:
      data/raw/MUVIM/ZED_RGB/ADL/NonFall10/*.png
      data/raw/MUVIM/ZED_RGB/Fall/Fall10/*.png
  and treat:
      "NonFallXX" → ADL
      "FallXX"    → FALL
- Missing detections are encoded as xy = NaN (33x2) and conf = 0 (33,).

Example
-------
URFD:
  python pose/extract_2d_from_images.py \
    --images_glob data/raw/UR_Fall/adl/*/*.png data/raw/UR_Fall/fall/*/*.png \
    --sequence_id_depth 1 \
    --out_dir data/interim/urfd/pose_npz \
    --dataset urfd \
    --label_component 2

CAUCAFall:
  python pose/extract_2d_from_images.py \
    --images_glob 'data/raw/CAUCAFall/CAUCAFall/*/*/*.png' \
    --sequence_id_depth 2 \
    --out_dir data/interim/caucafall/pose_npz \
    --dataset caucafall

MUVIM (ZED RGB):
  python pose/extract_2d_from_images.py \
    --images_glob 'data/raw/MUVIM/ZED_RGB/ADL/*/*.png' 'data/raw/MUVIM/ZED_RGB/Fall/*/*.png' \
    --sequence_id_depth 1 \
    --out_dir data/interim/muvim/pose_npz \
    --dataset muvim
"""

import os
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import cv2
import mediapipe as mp


# ---------------------------------------------------------------------
# Sequence grouping
# ---------------------------------------------------------------------
def list_sequences(image_globs, sequence_id_depth):
    """
    Group image file paths into sequences using `sequence_id_depth`.

    Parameters
    ----------
    image_globs : list[str] | str
        One or more glob patterns (quoted in the shell) that match image files.
        e.g., [".../adl/*/*.png", ".../fall/*/*.png"]
    sequence_id_depth : int
        How many directory components (above the filename) define a sequence.

    Returns
    -------
    dict[str, list[str]]
        Mapping from sequence ID (like "adl/seq_01") to a **sorted** list of frame paths.
    """
    frames_by_seq = defaultdict(list)
    paths = []

    if isinstance(image_globs, list):
        globs_list = image_globs
    else:
        globs_list = [image_globs]

    for g in globs_list:
        paths.extend(glob.glob(g, recursive=True))

    for f in sorted(paths):
        parts = os.path.normpath(f).split(os.sep)
        # Example: [..., "adl", "seq_01", "frame_0001.png"]
        seq_parts = parts[-(sequence_id_depth + 1):-1]
        seq_id = "/".join(seq_parts)
        frames_by_seq[seq_id].append(f)

    return frames_by_seq


# ---------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------
def infer_label_from_path(path, which_component_from_root=None, mapping=None, rule=None):
    """
    Robustly infer 'adl' / 'fall' from a path or sequence ID.

    Returns 'adl' or 'fall'.
    """
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)
    parts_lower = [p.lower() for p in parts]

    # 1) Fixed component (URFD case)
    if which_component_from_root is not None and 0 <= which_component_from_root < len(parts_lower):
        comp = parts_lower[which_component_from_root]
        if mapping and comp in mapping:
            return mapping[comp]
        if comp in ("adl", "fall"):
            return comp

    # 2) Special CAUCAFall heuristic: last component contains "fall" or not
    if rule == "caucafall_rule":
        last = parts_lower[-1]
        if "fall" in last:
            return "fall"
        return "adl"

    # 3) Special MUVIM heuristic: "NonFallXX" vs "FallXX"
    if rule == "muvim_rule":
        # sequence_id_depth=1 → seq_id like "NonFall10" or "Fall10"
        last = parts_lower[-1]  # e.g. "nonfall10", "fall10"
        if last.startswith("nonfall"):
            return "adl"
        if last.startswith("fall"):
            return "fall"
        # Fallback: scan all components for pure "fall"/"adl"
        for p in parts_lower:
            if p == "fall":
                return "fall"
            if p == "adl":
                return "adl"
        return "adl"

    # 4) Generic fallback: search entire path string
    lp = norm_path.lower()
    if "fall" in lp:
        return "fall"
    if "adl" in lp:
        return "adl"
    return "adl"  # safe default


def label_to_numeric(label: str) -> float:
    """'fall' → 1.0, 'adl' → 0.0, unknown → -1.0."""
    s = str(label).lower()
    if "fall" in s:
        return 1.0
    if "adl" in s:
        return 0.0
    return -1.0


# ---------------------------------------------------------------------
# Per-sequence extraction
# ---------------------------------------------------------------------
def extract_sequence(frames, pose, out_npz, label=None):
    """
    Run MediaPipe Pose on a list of frame paths and save one NPZ for the sequence.

    Output NPZ (compressed)
    -----------------------
    - xy  : float32 [T, 33, 2]  (normalized x,y; NaN for missing)
    - conf: float32 [T, 33]     (visibility; zeros for missing)
    - y   : float32 scalar      (1.0 fall / 0.0 adl / -1.0 unlabeled)
    """
    xy_list, conf_list = [], []

    for f in frames:
        img = cv2.imread(f)
        if img is None:
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
        else:
            lm = res.pose_landmarks.landmark
            xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)[:33]
            cf = np.array([p.visibility for p in lm], dtype=np.float32)[:33]
            xy_list.append(xy)
            conf_list.append(cf)

    if xy_list:
        xy = np.stack(xy_list)
        cf = np.stack(conf_list)
    else:
        xy = np.zeros((0, 33, 2), np.float32)
        cf = np.zeros((0, 33), np.float32)

    # numeric label (video-level)
    y_val = label_to_numeric(label) if label is not None else -1.0
    y_arr = np.array(y_val, dtype=np.float32)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, xy=xy, conf=cf, y=y_arr)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images_glob",
        nargs="+",
        required=True,
        help="One or more glob patterns for images. Quote patterns to avoid shell expansion.",
    )
    ap.add_argument(
        "--sequence_id_depth",
        type=int,
        required=True,
        help="How many parent directories above the filename define a sequence ID.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Where to save NPZ files (one per sequence).",
    )
    ap.add_argument(
        "--dataset",
        choices=["urfd", "caucafall", "muvim"],
        required=True,
        help="Enables dataset-specific labelling rules.",
    )
    ap.add_argument(
        "--label_component",
        type=int,
        default=None,
        help="URFD: 0-based index of path component to read label from (e.g., 'adl' or 'fall').",
    )
    args = ap.parse_args()

    # MediaPipe Pose model
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
    )

    # Group frames into sequences
    frames_by_seq = list_sequences(args.images_glob, args.sequence_id_depth)
    total_frames = sum(len(v) for v in frames_by_seq.values())
    print(f"[INFO] Found {len(frames_by_seq)} sequences, {total_frames} frames total")

    labels = {}  # npz_stem -> 'adl'/'fall'

    for i, (seq_id, frames) in enumerate(frames_by_seq.items(), start=1):
        print(f"[INFO] Sequence {i}/{len(frames_by_seq)}: {seq_id} ({len(frames)} frames)")

        if args.dataset == "urfd":
            # Label from a specific component (e.g. 'adl' or 'fall')
            lbl = infer_label_from_path(
                frames[0],
                which_component_from_root=args.label_component,
                mapping={"adl": "adl", "fall": "fall"},
            )
            out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")

        elif args.dataset == "caucafall":
            # Label from last component name (contains 'fall' or not)
            lbl = infer_label_from_path(seq_id, rule="caucafall_rule")
            out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")

        elif args.dataset == "muvim":
            # Use MUVIM-specific rule: NonFallXX → adl, FallXX → fall
            lbl = infer_label_from_path(seq_id, rule="muvim_rule")
            out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Run pose + save npz (xy, conf, y)
        extract_sequence(frames, mp_pose, out_npz, label=lbl)

        base = os.path.splitext(os.path.basename(out_npz))[0]
        labels[base] = lbl

    os.makedirs("configs/labels", exist_ok=True)
    labels_path = f"configs/labels/{args.dataset}_auto.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[OK] wrote {len(labels)} sequences and {labels_path}")
