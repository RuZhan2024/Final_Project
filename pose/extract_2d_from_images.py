#!/usr/bin/env python3
"""
Extract 2D human pose from **image sequences** (e.g., URFD/CAUCAFall frame folders)
using MediaPipe Pose and save **one NPZ per sequence**.

Each output NPZ stores:
  - xy  : float32 array of shape [T, 33, 2]  → normalized landmark coords (x, y in [0,1])
  - conf: float32 array of shape [T, 33]     → landmark visibility/confidence proxy

Notes
-----
- We group frames into a "sequence" using `sequence_id_depth` (how many parent dirs above
  the image filename to include in the sequence ID).
- For **URFD** we typically have paths like:
      data/raw/UR_Fall/adl/adl-01-cam0-rgb/adl-01-cam0-rgb-001.png
  with label in the directory name ("adl" / "fall").
- For **CAUCAFall** we infer labels by checking the last path component (sequence name)
  and looking for the substring "fall" (case-insensitive). Everything else is "adl".
- Missing detections are encoded as xy = NaN (33x2) and conf = 0 (33,). Downstream code
  can mask/fill NaNs before training, or let the windowing/loader handle them.

Example
-------
URFD:
  python pose/extract_2d_from_images.py \
    --images_glob data/raw/UR_Fall/adl/*/*.png data/raw/UR_Fall/fall/*/*.png \
    --sequence_id_depth 1 \
    --out_dir data/interim/urfd/pose_npz \
    --dataset urfd

CAUCAFall:
  python pose/extract_2d_from_images.py \
    --images_glob 'data/raw/CAUCAFall/CAUCAFall/*/*/*.png' \
    --sequence_id_depth 2 \
    --out_dir data/interim/caucafall/pose_npz \
    --dataset caucafall
"""

import os
import argparse
import glob
import json
import re
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
        For example, with path ".../adl/seq_01/frame_0001.png":
          - depth=1 → seq_id = "seq_01"
          - depth=2 → seq_id = "adl/seq_01"

    Returns
    -------
    dict[str, list[str]]
        Mapping from sequence ID (like "adl/seq_01") to a **sorted** list of frame paths.
    """
    frames_by_seq = defaultdict(list)
    paths = []

    # Expand one or many globs into a flat list of files
    if isinstance(image_globs, list):
        globs = image_globs
    else:
        globs = [image_globs]

    for g in globs:
        paths.extend(glob.glob(g, recursive=True))

    # Sort for deterministic order; group by last `sequence_id_depth` directories
    for f in sorted(paths):
        parts = os.path.normpath(f).split(os.sep)
        # Example: parts = [..., "adl", "seq_01", "frame_0001.png"]
        # sequence_id_depth=2 → ["adl", "seq_01"]
        seq_parts = parts[-(sequence_id_depth + 1):-1]
        seq_id = "/".join(seq_parts)  # normalized, portable ID like "adl/seq_01"
        frames_by_seq[seq_id].append(f)

    return frames_by_seq


# ---------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------
def infer_label_from_path(path, which_component_from_root=None, mapping=None, rule=None):
    """
    Robustly infer 'adl' / 'fall' from a path.

    Parameters
    ----------
    path : str
        File path or sequence ID.
    which_component_from_root : int | None
        If not None, try to read label from this component of the path (0-based).
    mapping : dict | None
        Optional mapping from component string → canonical label, e.g. {"adl": "adl", "fall": "fall"}.
    rule : str | None
        Extra rule selector; currently supports "caucafall_rule".

    Returns
    -------
    str
        'adl' or 'fall'
    """
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)
    parts_lower = [p.lower() for p in parts]

    # 1) Use specific component if requested (URFD case)
    if which_component_from_root is not None and 0 <= which_component_from_root < len(parts_lower):
        comp = parts_lower[which_component_from_root]
        if mapping and comp in mapping:
            return mapping[comp]
        if comp in ("adl", "fall"):
            return comp

    # 2) Special CAUCAFall heuristic: look at last component
    if rule == "caucafall_rule":
        last = parts_lower[-1]
        if "fall" in last:
            return "fall"
        return "adl"

    # 3) Generic fallback: search entire path string
    lp = norm_path.lower()
    if "fall" in lp:
        return "fall"
    if "adl" in lp:
        return "adl"
    return "adl"  # safe default


# ---------------------------------------------------------------------
# Per-sequence extraction
# ---------------------------------------------------------------------
def extract_sequence(frames, pose, out_npz):
    """
    Run MediaPipe Pose on a list of frame paths and save one NPZ for the sequence.

    Parameters
    ----------
    frames : list[str]
        Sorted list of image file paths belonging to the same sequence.
    pose : mp.solutions.pose.Pose
        A constructed MediaPipe Pose object (reused for performance).
    out_npz : str
        Destination NPZ path. Parent directories are created if missing.

    Output NPZ (compressed)
    -----------------------
    - xy  : float32 [T, 33, 2]  (normalized x,y; NaN for missing detections)
    - conf: float32 [T, 33]     (visibility; zeros for missing detections)
    """
    xy_list, conf_list = [], []

    for f in frames:
        img = cv2.imread(f)
        if img is None:
            # Could not read this frame (corrupt/missing) → mark all joints missing.
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
            continue

        # OpenCV loads BGR; MediaPipe expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process single image
        res = pose.process(rgb)

        if not res.pose_landmarks:
            # No person detected in this frame
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
        else:
            lm = res.pose_landmarks.landmark
            # Extract normalized landmark coordinates and visibility (first 33 points).
            xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)[:33]
            cf = np.array([p.visibility for p in lm], dtype=np.float32)[:33]
            xy_list.append(xy)
            conf_list.append(cf)

    # Stack along time T. If there were no valid frames, create empty arrays with correct shape.
    if xy_list:
        xy = np.stack(xy_list)
        cf = np.stack(conf_list)
    else:
        xy = np.zeros((0, 33, 2), np.float32)
        cf = np.zeros((0, 33), np.float32)

    # Ensure the parent directory exists and write compressed NPZ
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, xy=xy, conf=cf)


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
        choices=["urfd", "caucafall"],
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

    # Construct a single Pose model and reuse across all frames/sequences (faster).
    # model_complexity=1 is a speed/accuracy trade-off; use 2 for highest accuracy.
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
    )

    # Group frame paths into sequences according to the requested depth
    frames_by_seq = list_sequences(args.images_glob, args.sequence_id_depth)
    total_frames = sum(len(v) for v in frames_by_seq.values())
    print(f"[INFO] Found {len(frames_by_seq)} sequences, {total_frames} frames total")

    labels = {}  # seq_stem -> 'adl'/'fall' mapping to be written into configs/labels/*.json

    # Process each sequence
    for i, (seq_id, frames) in enumerate(frames_by_seq.items(), start=1):
        print(f"[INFO] Sequence {i}/{len(frames_by_seq)}: {seq_id} ({len(frames)} frames)")

        # Decide dataset-specific labelling and output path convention.
        if args.dataset == "urfd":
            lbl = infer_label_from_path(
                frames[0],
                which_component_from_root=args.label_component,
                mapping={"adl": "adl", "fall": "fall"},
            )
            out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")
        else:  # caucafall
            lbl = infer_label_from_path(seq_id, rule="caucafall_rule")
            out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")

        # Run pose on all frames in this sequence and write the NPZ file
        extract_sequence(frames, mp_pose, out_npz)

        # Store the label under the NPZ basename (without ".npz"); used later by windowing
        base = os.path.splitext(os.path.basename(out_npz))[0]
        labels[base] = lbl

    # Persist the auto-generated labels mapping so downstream steps can use it.
    os.makedirs("configs/labels", exist_ok=True)
    labels_path = f"configs/labels/{args.dataset}_auto.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[OK] wrote {len(labels)} sequences and {labels_path}")
