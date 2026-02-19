#!/usr/bin/env python3
"""
Extract 2D human pose from image sequences using MediaPipe Pose and save one NPZ per sequence.

Output NPZ keys (pose-only + metadata):
  - xy   : float32 [T, 33, 2]  normalized coords (0..1); 0 when missing
  - conf : float32 [T, 33]     visibility/confidence; 0 when missing
  - fps  : float32
  - size : int32   [2]         [width, height] from the first readable frame (0,0 if unknown)
  - src  : str                 representative source directory (sequence folder)
  - seq_id : str               the sequence id used for grouping

Notes
-----
- This script is label-agnostic by design (labels come later).
- Sequences are grouped by `sequence_id_depth` directories above the filename.
"""

import os
import re
import glob
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import mediapipe as mp

J = 33  # MediaPipe Pose landmarks

DEFAULT_FPS = {
    "urfd": 30.0,
    "caucafall": 23.0,
    "muvim": 30.0,
}

# Silence verbose logs BEFORE importing mediapipe backends (best effort)
os.environ.setdefault("GLOG_minloglevel", "2")     # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _natural_key(s: str):
    """Natural sort key: frame_2.png < frame_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sanitize_stem(stem: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


def list_sequences(image_globs, sequence_id_depth: int):
    """
    Return:
      frames_by_seq: dict[seq_id, list[frame_path]]
      all_paths: list[str]
    """
    frames_by_seq = defaultdict(list)
    all_paths = []

    globs_list = image_globs if isinstance(image_globs, list) else [image_globs]

    for g in globs_list:
        all_paths.extend(glob.glob(g, recursive=True))

    all_paths = [p for p in all_paths if os.path.isfile(p)]
    all_paths.sort(key=_natural_key)

    for f in all_paths:
        parts = os.path.normpath(f).split(os.sep)
        if sequence_id_depth <= 0:
            # whole parent folder becomes seq id
            seq_parts = parts[-2:-1]
        else:
            # take N dirs above filename
            if len(parts) < sequence_id_depth + 1:
                seq_parts = parts[:-1]
            else:
                seq_parts = parts[-(sequence_id_depth + 1):-1]
        seq_id = "/".join(seq_parts) if seq_parts else "sequence"
        frames_by_seq[seq_id].append(f)

    return dict(frames_by_seq), all_paths


def _safe_out_name(seq_id: str, src_rel: str) -> str:
    """
    Stable filename from seq_id + short hash of *relative* source directory.

    Using a relative path keeps filenames reproducible across machines.
    """
    base = _sanitize_stem(seq_id.replace("/", "__"))
    src_rel_norm = src_rel.replace(os.sep, "/")
    h = hashlib.md5(src_rel_norm.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}.npz"


def extract_sequence(frames, pose, out_npz: str, fps: float, src: str, seq_id: str, store_frames: bool = False):
    """
    Save:
      xy  [T,33,2] (0 when missing)
      conf[T,33]
      fps, size, src, seq_id
    """
    xy_list, conf_list = [], []
    size = np.array([0, 0], dtype=np.int32)  # [w,h]

    for f in frames:
        img = cv2.imread(f)
        if img is None:
            xy_list.append(np.zeros((J, 2), np.float32))
            conf_list.append(np.zeros((J,), np.float32))
            continue

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

    if xy_list:
        xy = np.stack(xy_list, axis=0)
        cf = np.stack(conf_list, axis=0)
    else:
        xy = np.zeros((0, J, 2), np.float32)
        cf = np.zeros((0, J), np.float32)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        xy=xy.astype(np.float32),
        conf=cf.astype(np.float32),
        fps=np.float32(fps),
        size=size,
        src=np.str_(src),
        seq_id=np.str_(seq_id),
    )
    if store_frames:
        payload["frames"] = np.array(frames, dtype=np.str_)

    # Atomic write: avoid partially-written files being "skipped" forever.
    tmp_npz = out_npz + ".tmp.npz"
    np.savez_compressed(tmp_npz, **payload)
    os.replace(tmp_npz, out_npz)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract 2D pose (MediaPipe) from image sequences into per-sequence NPZs."
    )
    ap.add_argument("--images_glob", nargs="+", required=True,
                    help="One or more glob patterns for images. Quote patterns in shell.")
    ap.add_argument("--sequence_id_depth", type=int, required=True,
                    help="How many parent directories above filename define a sequence.")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write NPZ files (one per sequence).")
    ap.add_argument("--dataset", choices=["urfd", "caucafall", "muvim"], default=None,
                    help="Used only to pick a default FPS if --fps is not provided.")
    ap.add_argument("--fps", type=float, default=None,
                    help="FPS to store. If omitted uses dataset default or 30.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip if the output NPZ already exists.")
    ap.add_argument("--max_sequences", type=int, default=None,
                    help="Optional cap for debugging (process only first N sequences).")
    ap.add_argument("--store_frames", action="store_true",
                    help="Store frame paths in NPZ (bigger files).")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                    help="0=lite,1=full,2=heavy (default 1).")
    ap.add_argument("--min_det_conf", type=float, default=0.5,
                    help="Min detection confidence.")
    ap.add_argument("--min_track_conf", type=float, default=0.5,
                    help="Min tracking confidence.")
    ap.add_argument("--static_image_mode", action="store_true",
                    help="If set, disables temporal tracking (usually slower for sequences).")
    # kept for backwards compatibility (ignored)
    ap.add_argument("--label_component", type=int, default=None,
                    help="(Deprecated/ignored) Previously used for inline labelling.")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.fps is not None:
        fps_value = float(args.fps)
    elif args.dataset in DEFAULT_FPS:
        fps_value = float(DEFAULT_FPS[args.dataset])
    else:
        fps_value = 30.0

    frames_by_seq, all_paths = list_sequences(args.images_glob, args.sequence_id_depth)
    if not frames_by_seq:
        raise SystemExit("[ERR] no images matched your glob(s).")

    # common root for readable src dirs (best effort)
    try:
        common_root = os.path.commonpath(all_paths)
    except Exception:
        common_root = os.path.dirname(all_paths[0])

    seq_items = list(frames_by_seq.items())
    if args.max_sequences is not None:
        seq_items = seq_items[: max(0, int(args.max_sequences))]

    print(f"[INFO] Found {len(frames_by_seq)} sequences, using {len(seq_items)} sequences")

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as pose:

        for i, (seq_id, frames) in enumerate(seq_items, start=1):
            seq_src_dir = os.path.commonpath(frames) if frames else ""
            try:
                src_rel = os.path.relpath(seq_src_dir, common_root)
            except Exception:
                src_rel = seq_src_dir

            out_name = _safe_out_name(seq_id, src_rel)
            out_npz = os.path.join(args.out_dir, out_name)

            if args.skip_existing and os.path.exists(out_npz):
                print(f"[skip] {i}/{len(seq_items)} exists: {out_name}")
                continue

            print(f"[pose] {i}/{len(seq_items)} seq_id={seq_id}  frames={len(frames)}  src={src_rel}")
            extract_sequence(
                frames,
                pose,
                out_npz,
                fps=fps_value,
                src=src_rel,
                seq_id=seq_id,
                store_frames=args.store_frames,
            )

    print(f"[OK] wrote {len(seq_items)} sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
