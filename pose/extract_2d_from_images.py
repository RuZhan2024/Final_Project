#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose/extract_2d_from_images.py

Extract 2D human pose from IMAGE SEQUENCES using MediaPipe Pose and save one NPZ per sequence.

Output NPZ keys:
  - xy     : float32 [T, 33, 2]  normalized coords (0..1); zeros when missing
  - conf   : float32 [T, 33]     visibility/confidence; zeros when missing
  - fps    : float32             stored fps (dataset default or --fps)
  - size   : int32   [2]         [width, height] from the first readable frame (0,0 if unknown)
  - src    : str                 representative source directory (sequence folder)
  - seq_id : str                 sequence id derived from directory depth
  - frames : (optional) str [T]  frame paths (if --store_frames is enabled)

Notes:
- This script is label-agnostic by design (labels come later).
- Sequences are grouped by `sequence_id_depth` directories above the filename.
"""

from __future__ import annotations

# ============================================================
# 0) Silence backend logs BEFORE importing mediapipe.
# ============================================================
import os as _os

_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# 1) Imports
# ============================================================
import argparse
import glob
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import tempfile

import cv2
import mediapipe as mp
import numpy as np

J = 33  # MediaPipe Pose landmarks

# Some datasets have known FPS defaults for image sequences.
DEFAULT_FPS = {
    "urfd": 30.0,
    "caucafall": 23.0,
    "muvim": 30.0,
}


# ============================================================
# 2) Utility helpers
# ============================================================
def _natural_key(s: str):
    """
    Natural sort key:
      frame_2.png comes before frame_10.png

    Why:
    - Image sequences are often numbered, and normal lexicographic sort is wrong.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sanitize_stem(stem: str) -> str:
    """
    Convert an arbitrary string into a filename-safe stem.
    """
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


def _atomic_save_npz(out_npz: str, payload: dict) -> None:
    """
    Atomic NPZ write:
    - write to a temp file (must end with .npz)
    - replace final path atomically

    Why we must force ".npz" on temp filenames:
    - If temp filename is "x.npz.tmp", numpy may write "x.npz.tmp.npz"
      and then os.replace("x.npz.tmp", "x.npz") fails.
    """
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.stem + ".",
        suffix=".tmp.npz",          # MUST end with .npz
        dir=str(out_path.parent),
    )
    os.close(fd)

    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, str(out_path))
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def _safe_out_name(seq_id: str, seq_src_dir: str) -> str:
    """
    Collision-safe filename from seq_id + short hash of source directory.

    Why:
    - Different sequences can end up with the same seq_id string
      (e.g., if folder layouts differ).
    - Hash makes collisions extremely unlikely.
    """
    base = _sanitize_stem(seq_id.replace("/", "__"))
    h = hashlib.md5(seq_src_dir.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}.npz"


# ============================================================
# 3) Sequence grouping
# ============================================================
def list_sequences(image_globs: Sequence[str], sequence_id_depth: int) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Group image file paths into sequences.

    Inputs:
      image_globs:
        One or more glob patterns (recursive patterns supported).
      sequence_id_depth:
        How many parent directories above filename define a sequence.

    Returns:
      frames_by_seq: dict { seq_id -> [frame paths] }
      all_paths: all matched file paths (flat list)

    Example:
      If sequence_id_depth=2 and path is:
        dataset/cam01/run07/frame_0001.png
      then seq_id becomes:
        cam01/run07
    """
    frames_by_seq: Dict[str, List[str]] = defaultdict(list)
    all_paths: List[str] = []

    for g in image_globs:
        all_paths.extend(glob.glob(g, recursive=True))

    # Keep only files, sorted naturally.
    all_paths = [p for p in all_paths if os.path.isfile(p)]
    all_paths.sort(key=_natural_key)

    for f in all_paths:
        p = Path(f)

        # Build seq_parts from parent directories.
        # parents[0] is immediate parent; parents[1] is grandparent; etc.
        parents = [pp.name for pp in p.parents if pp.name]  # includes many levels

        if sequence_id_depth <= 0:
            # Depth <= 0 means "use only the immediate parent folder".
            seq_parts = [p.parent.name] if p.parent.name else ["sequence"]
        else:
            # Take N directories above filename, from outer->inner order.
            # Example: depth=2 -> [grandparent, parent]
            take = parents[:sequence_id_depth]  # from inner outward
            seq_parts = list(reversed(take)) if take else ["sequence"]

        seq_id = "/".join(seq_parts) if seq_parts else "sequence"
        frames_by_seq[seq_id].append(str(p))

    return dict(frames_by_seq), all_paths


# ============================================================
# 4) Pose extraction for one sequence
# ============================================================
def extract_sequence(
    frames: List[str],
    pose,
    *,
    out_npz: str,
    fps: float,
    seq_id: str,
    seq_src_dir: str,
    store_frames: bool = False,
) -> None:
    """
    Extract pose for all frames in one sequence and save NPZ.

    Important outputs:
    - xy  [T,33,2] normalized coords in [0,1]
    - conf[T,33]   landmark visibility
    - size [w,h]   from the first readable image
    - src, seq_id  metadata strings

    Behavior when a frame cannot be read / no pose detected:
    - fill xy/conf with zeros for that frame (keeps time alignment stable)
    """
    xy_list: List[np.ndarray] = []
    conf_list: List[np.ndarray] = []

    # [w,h] in pixels. If we never read a valid image, it stays [0,0].
    size = np.array([0, 0], dtype=np.int32)

    for f in frames:
        img = cv2.imread(f)
        if img is None:
            # Frame missing/corrupt -> keep alignment by appending zeros.
            xy_list.append(np.zeros((J, 2), np.float32))
            conf_list.append(np.zeros((J,), np.float32))
            continue

        # Set size once based on the first readable image.
        if size[0] == 0 and size[1] == 0:
            h, w = img.shape[:2]
            size = np.array([w, h], dtype=np.int32)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            xy_list.append(np.zeros((J, 2), np.float32))
            conf_list.append(np.zeros((J,), np.float32))
        else:
            lm = res.pose_landmarks.landmark
            xy = np.array([[p.x, p.y] for p in lm[:J]], dtype=np.float32)
            cf = np.array([p.visibility for p in lm[:J]], dtype=np.float32)
            xy_list.append(xy)
            conf_list.append(cf)

    # Stack into arrays (T is number of frames)
    if xy_list:
        xy = np.stack(xy_list, axis=0).astype(np.float32)
        cf = np.stack(conf_list, axis=0).astype(np.float32)
    else:
        xy = np.zeros((0, J, 2), np.float32)
        cf = np.zeros((0, J), np.float32)

    payload = dict(
        xy=xy,
        conf=cf,
        fps=np.float32(float(fps)),
        size=size,
        src=np.str_(seq_src_dir),
        seq_id=np.str_(seq_id),
    )

    if store_frames:
        payload["frames"] = np.array(frames, dtype=np.str_)

    _atomic_save_npz(out_npz, payload)


# ============================================================
# 5) CLI + main
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract 2D pose (MediaPipe) from image sequences into per-sequence NPZs."
    )

    ap.add_argument(
        "--images_glob",
        nargs="+",
        required=True,
        help="One or more glob patterns for images. Quote patterns in shell.",
    )
    ap.add_argument(
        "--sequence_id_depth",
        type=int,
        required=True,
        help="How many parent directories above filename define a sequence.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Where to write NPZ files (one per sequence).",
    )

    ap.add_argument(
        "--dataset",
        choices=["urfd", "caucafall", "muvim"],
        default=None,
        help="Used only to pick a default FPS if --fps is not provided.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS to store. If omitted uses dataset default or 30.",
    )

    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if the output NPZ already exists.",
    )
    ap.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optional cap for debugging (process only first N sequences).",
    )
    ap.add_argument(
        "--store_frames",
        action="store_true",
        help="Store frame paths in NPZ (bigger files).",
    )

    ap.add_argument(
        "--model_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=lite,1=full,2=heavy (default 1).",
    )
    ap.add_argument(
        "--min_det_conf",
        type=float,
        default=0.5,
        help="Min detection confidence.",
    )
    ap.add_argument(
        "--min_track_conf",
        type=float,
        default=0.5,
        help="Min tracking confidence.",
    )
    ap.add_argument(
        "--static_image_mode",
        action="store_true",
        help="If set, treat each frame independently (slower; usually worse for sequences).",
    )

    # kept for backwards compatibility (ignored)
    ap.add_argument(
        "--label_component",
        type=int,
        default=None,
        help="(Deprecated/ignored) Previously used for inline labelling.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Decide the FPS value we store in NPZ
    if args.fps is not None:
        fps_value = float(args.fps)
    elif args.dataset in DEFAULT_FPS:
        fps_value = float(DEFAULT_FPS[args.dataset])
    else:
        fps_value = 30.0

    frames_by_seq, all_paths = list_sequences(args.images_glob, args.sequence_id_depth)
    if not frames_by_seq:
        raise SystemExit("[ERR] no images matched your glob(s).")

    # We compute a common root only for prettier logging (best effort).
    try:
        common_root = os.path.commonpath(all_paths)
    except Exception:
        common_root = str(Path(all_paths[0]).parent)

    seq_items = list(frames_by_seq.items())
    if args.max_sequences is not None:
        seq_items = seq_items[: max(0, int(args.max_sequences))]

    print(f"[INFO] Found {len(frames_by_seq)} sequences, using {len(seq_items)} sequences")

    mp_pose = mp.solutions.pose

    # Create one Pose instance and reuse it (faster).
    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as pose:

        for i, (seq_id, frames) in enumerate(seq_items, start=1):
            # seq_src_dir = best guess of the directory that represents this sequence
            seq_src_dir = os.path.commonpath(frames) if frames else ""

            # A relative source string (only for logs)
            try:
                src_rel = os.path.relpath(seq_src_dir, common_root)
            except Exception:
                src_rel = seq_src_dir

            out_name = _safe_out_name(seq_id, seq_src_dir)
            out_npz = os.path.join(args.out_dir, out_name)

            if args.skip_existing and os.path.exists(out_npz):
                print(f"[skip] {i}/{len(seq_items)} exists: {out_name}")
                continue

            print(f"[pose] {i}/{len(seq_items)} seq_id={seq_id}  frames={len(frames)}  src={src_rel}")

            extract_sequence(
                frames,
                pose,
                out_npz=out_npz,
                fps=fps_value,
                seq_id=seq_id,
                seq_src_dir=seq_src_dir,
                store_frames=args.store_frames,
            )

    print(f"[OK] wrote {len(seq_items)} sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
