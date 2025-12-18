
"""
Extract 2D human pose from **image sequences** (e.g., URFD/CAUCAFall/MUVIM frame folders)
using MediaPipe Pose and save **one NPZ per sequence**.

Canonical pipeline (for all datasets):
  1) Pose extraction (this script) → unlabeled per-sequence NPZ files
  2) Dataset-specific label scripts → configs/labels/<dataset>.json
  3) make_splits.py                → train/val/test splits (by sequence stem)
  4) make_windows.py               → sliding windows with label y=0/1 per window
  5) train_tcn.py / train_gcn.py   → train classifiers on window NPZs

Each output NPZ from this script stores ONLY pose features:

  - xy  : float32 [T, 33, 2]  → normalized landmark coords (x, y in [0,1])
  - conf: float32 [T, 33]     → landmark visibility/confidence proxy

Notes
-----
- We group frames into a "sequence" using `sequence_id_depth`
  (how many parent dirs above the image filename to include in the sequence ID).
- This script is deliberately **label-agnostic**. URFD / CAUCAFall / MUVIM labels
  should be handled by separate dataset-specific scripts that read these NPZs
  and/or their stems and produce a labels JSON.
"""

import os
import argparse
import glob
from collections import defaultdict

import numpy as np
import cv2
import mediapipe as mp


DEFAULT_FPS = {
    "urfd": 30.0,
    "caucafall": 23.0,
    "muvim": 30.0,
}

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

        Example
        -------
        path = data/raw/UR_Fall/adl/adl-01-cam0-rgb/frame_0001.png
        sequence_id_depth = 1
          → seq_id = "adl-01-cam0-rgb"
        sequence_id_depth = 2
          → seq_id = "adl/adl-01-cam0-rgb"

    Returns
    -------
    dict[str, list[str]]
        Mapping from sequence ID (like "adl/seq_01") to a **sorted** list of
        frame paths that belong to that sequence.
    """
    frames_by_seq = defaultdict(list)
    paths = []

    # Allow a single glob string or a list of globs
    if isinstance(image_globs, list):
        globs_list = image_globs
    else:
        globs_list = [image_globs]

    # Expand all glob patterns to actual file paths
    for g in globs_list:
        paths.extend(glob.glob(g, recursive=True))

    # Sort for deterministic ordering and then group into sequences
    for f in sorted(paths):
        parts = os.path.normpath(f).split(os.sep)
        # Example: [..., "adl", "seq_01", "frame_0001.png"]
        # We take `sequence_id_depth` directories above the filename
        seq_parts = parts[-(sequence_id_depth + 1):-1]
        seq_id = "/".join(seq_parts)
        frames_by_seq[seq_id].append(f)

    return frames_by_seq


# ---------------------------------------------------------------------
# Per-sequence extraction (pose only)
# ---------------------------------------------------------------------
def extract_sequence(frames, pose, out_npz, fps: float):
    """
    Run MediaPipe Pose on a list of frame paths and save one NPZ for the sequence.

    Output NPZ (compressed)
    -----------------------
    - xy  : float32 [T, 33, 2]  (normalized x,y; NaN for missing)
    - conf: float32 [T, 33]     (visibility; zeros for missing)

    This script does **not** attach labels. Labels are handled later by
    dataset-specific scripts that look at directory structure, filenames,
    or external annotation files.
    """
    xy_list, conf_list = [], []

    for f in frames:
        # Read image from disk (BGR format by default in OpenCV)
        img = cv2.imread(f)
        if img is None:
            # If the image cannot be read, mark this frame as "missing":
            #   xy   → NaNs
            #   conf → zeros
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
            continue

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Run pose estimation on this frame
        res = pose.process(rgb)

        if not res.pose_landmarks:
            # No pose found in this frame → store NaNs and zero confidence
            xy_list.append(np.full((33, 2), np.nan, np.float32))
            conf_list.append(np.zeros((33,), np.float32))
        else:
            lm = res.pose_landmarks.landmark
            # Take up to 33 landmarks → [T, 33, 2]
            xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)[:33]
            # Visibility as per-joint confidence proxy → [T, 33]
            cf = np.array([p.visibility for p in lm], dtype=np.float32)[:33]
            xy_list.append(xy)
            conf_list.append(cf)

    # Stack into [T, 33, 2] and [T, 33]
    if xy_list:
        xy = np.stack(xy_list)
        cf = np.stack(conf_list)
    else:
        # Empty sequence edge case (shouldn't normally happen)
        xy = np.zeros((0, 33, 2), np.float32)
        cf = np.zeros((0, 33), np.float32)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    # Save *pose only* (xy + conf); no labels here by design
    np.savez_compressed(out_npz, xy=xy, conf=cf, fps=np.float32(fps))


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract 2D pose (MediaPipe) from image sequences into per-sequence NPZs."
    )
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
    # NOTE: dataset / label_component are no longer needed for extraction itself,
    # but if you want to keep CLI compatible with old scripts, you can either:
    #  - keep them as dummy options and ignore them, or
    #  - remove them and update your Makefile/commands.
    ap.add_argument(
        "--dataset",
        choices=["urfd", "caucafall", "muvim"],
        required=False,
        help="(Deprecated) Previously used for inline labelling. Ignored in this pose-only version.",
    )
    ap.add_argument(
        "--label_component",
        type=int,
        default=None,
        help="(Deprecated) Previously used for inline labelling. Ignored in this pose-only version.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS to store in output NPZs. If omitted, uses a dataset default (urfd=30, caucafall=23, muvim=30).",
    )
    args = ap.parse_args()

    # Determine FPS to store for this dataset
    if getattr(args, "fps", None) is not None:
        fps_value = float(args.fps)
    elif args.dataset in DEFAULT_FPS:
        fps_value = float(DEFAULT_FPS[args.dataset])
    else:
        fps_value = 30.0

    # Initialise MediaPipe Pose model (same settings you use elsewhere)
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
    )

    # Group individual frames into sequences
    frames_by_seq = list_sequences(args.images_glob, args.sequence_id_depth)
    total_frames = sum(len(v) for v in frames_by_seq.values())
    print(f"[INFO] Found {len(frames_by_seq)} sequences, {total_frames} frames total")

    for i, (seq_id, frames) in enumerate(frames_by_seq.items(), start=1):
        print(f"[INFO] Sequence {i}/{len(frames_by_seq)}: {seq_id} ({len(frames)} frames)")
        # One NPZ per sequence; replace "/" in the sequence id to keep filenames safe
        out_npz = os.path.join(args.out_dir, seq_id.replace("/", "_") + ".npz")
        extract_sequence(frames, mp_pose, out_npz, fps=fps_value)

    print(f"[OK] wrote {len(frames_by_seq)} sequences to {args.out_dir}")
