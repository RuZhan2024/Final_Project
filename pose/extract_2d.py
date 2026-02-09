#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose/extract_2d.py

Extract 2D human pose from VIDEOS using MediaPipe Pose and save one .npz per video.

Each output .npz contains:
  - xy   : float32 [T, 33, 2]   normalized landmark coordinates in [0,1]
  - conf : float32 [T, 33]      landmark visibility/confidence in [0,1]
  - fps  : float32              FPS stored for this video (from metadata or fallback)
  - size : int32   [2]          [width, height] in pixels
  - src  : str                  original video path

Design notes:
- This script is *label-agnostic* (labels are added later by labels/build scripts).
- It produces a "pose_npz" stage output that later preprocessing/windowing scripts consume.

Typical usage:
  python pose/extract_2d.py \
    --videos_glob 'data/raw/LE2i/**/videos/*.mp4' 'data/raw/LE2i/**/*.avi' \
    --out_dir 'data/interim/le2i/pose_npz'
"""

from __future__ import annotations

# ============================================================
# 0) Silence noisy backend logs BEFORE importing mediapipe.
# ============================================================
# MediaPipe uses TF/C++ logs in some builds.
# Setting these early reduces console spam and makes training logs readable.
import os as _os

_os.environ.setdefault("GLOG_minloglevel", "2")      # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=info,2=warning,3=error

# ============================================================
# 1) Imports
# ============================================================
import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import tempfile
import re
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose outputs 33 landmarks by default.
J = 33


# ============================================================
# 2) CLI arguments (each option is documented for learning)
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Keep this function small and explicit:
    - It's easier to understand and easier to debug.
    """
    ap = argparse.ArgumentParser(
        description="Extract 2D pose (MediaPipe) from videos into per-video NPZs."
    )

    ap.add_argument(
        "--videos_glob",
        nargs="+",
        required=True,
        help=(
            "One or more quoted glob patterns for videos. "
            "Example: '.../*.mp4' '.../*.avi'. "
            "We expand globs in Python so recursive patterns work consistently."
        ),
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory where output NPZ files will be written (one NPZ per video).",
    )

    # MediaPipe Pose model complexity:
    # - 0: lightest and fastest
    # - 1: default (good quality, moderate speed)
    # - 2: heaviest (best quality, slowest)
    ap.add_argument(
        "--model_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose complexity: 0=lite, 1=full (default), 2=heavy.",
    )

    # Detection vs tracking confidence:
    # - detection confidence controls finding a body from scratch
    # - tracking confidence controls following a body across frames
    ap.add_argument(
        "--min_det_conf",
        type=float,
        default=0.5,
        help="Minimum detection confidence (0..1). Higher = fewer false detections, but may miss some frames.",
    )
    ap.add_argument(
        "--min_track_conf",
        type=float,
        default=0.5,
        help="Minimum tracking confidence (0..1). Higher = more stable tracking, but may drop when motion is fast.",
    )

    # For videos, static_image_mode usually should be False
    # (it enables temporal smoothing/tracking).
    ap.add_argument(
        "--static_image_mode",
        action="store_true",
        help="If set, disables temporal tracking (treats each frame like a separate image). Slower and usually worse for video.",
    )

    ap.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional debugging limit: process only the first N matched videos.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip processing when the output NPZ already exists.",
    )

    # FPS handling:
    # - cap.get(CAP_PROP_FPS) can return 0 or NaN for some files
    # - fps_default is the fallback when metadata is missing
    ap.add_argument(
        "--fps_default",
        type=float,
        default=30.0,
        help="Fallback FPS if video metadata FPS is missing/0 (default: 30).",
    )

    # force_fps:
    # - If your dataset has wrong FPS metadata, you can override it here.
    ap.add_argument(
        "--force_fps",
        type=float,
        default=None,
        help="If set, store this FPS for all videos (useful to correct dataset-wide metadata problems).",
    )

    ap.add_argument(
        "--log_every_s",
        type=float,
        default=3.0,
        help="Print progress at most once every N seconds per video (default: 3s). Set 0 to disable.",
    )

    return ap.parse_args()


# ============================================================
# 3) File discovery helpers
# ============================================================
def list_videos(patterns: Iterable[str]) -> List[str]:
    """
    Expand glob patterns, deduplicate results, and sort deterministically.

    The reason of doing this:
    - globbing behavior can vary slightly across shells/OS.
    - deterministic ordering makes it runs reproducible.
    """
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    # Keep only real files (filter out directories).
    files = [f for f in files if os.path.isfile(f)]

    # Define the natural sort key
    def natural_keys(text):
        # Splits text into a list of strings and integers:
        # "video (10).avi" -> ["video (", 10, ").avi"]
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    # Use the key to sort
    print(sorted(set(files), key=natural_keys))
    return sorted(set(files), key=natural_keys)


def common_root(paths: List[str]) -> str:
    """
    Best-effort shared root directory of a list of paths.
    This is used to build readable output filenames.
    """
    if not paths:
        return "."
    if len(paths) == 1:
        return os.path.dirname(paths[0])

    try:
        return os.path.commonpath(paths)
    except ValueError:
        # Happens if paths are on different drives (Windows) or are incompatible.
        return os.path.dirname(paths[0])


def make_safe_stem(video_path: str, root: str) -> str:
    """
    Convert a video path into a safe output filename stem.

    We encode the relative path into the filename to avoid collisions:
      root/.../Coffee_room_01/videos/fall01.mp4
        -> Coffee_room_01__videos__fall01

    This avoids the classic bug: many datasets reuse names like 'video01.mp4'.
    """
    rel = os.path.relpath(video_path, root)

    # Drop extension, replace path separators with "__"
    stem = os.path.splitext(rel)[0].replace(os.sep, "__")

    # Keep only safe filename characters
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


# ============================================================
# 4) Core extraction for one video
# ============================================================
def _atomic_save_npz(out_npz: str, payload: dict) -> None:
    """
    Atomically write NPZ:
    - write to a temporary file in the SAME directory
    - then rename to the final path using os.replace (atomic on same filesystem)
    """
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a unique temp file inside the same folder (atomic rename requires same filesystem).
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.stem + ".",
        suffix=".tmp.npz",          # MUST end with .npz
        dir=str(out_path.parent),
    )
    os.close(fd)

    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, str(out_path))  # atomic rename
    except Exception:
        # Best-effort cleanup if something failed mid-write.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def run_one_video(
    path: str,
    out_npz: str,
    pose,
    *,
    fps_default: float = 30.0,
    force_fps: Optional[float] = None,
    log_every_s: float = 3.0,
) -> bool:
    """
    Read a video frame-by-frame, run MediaPipe Pose, save results.

    Returns:
      True if extraction succeeded, False if the video couldn't be processed.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] cannot open: {path}", file=sys.stderr)
        return False

    # FPS from metadata is often unreliable; we clean it here.
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if force_fps is not None:
        fps = float(force_fps)
    elif (not fps) or np.isnan(fps) or fps <= 0.0:
        fps = float(fps_default)

    # Video size in pixels (can be 0 if metadata missing, but usually present)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_xy: List[np.ndarray] = []
    frames_conf: List[np.ndarray] = []

    frame_idx = 0
    last_log = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # MediaPipe expects RGB input (OpenCV loads BGR).
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # If conversion fails, append a "missing pose" frame.
            frames_xy.append(np.zeros((J, 2), dtype=np.float32))
            frames_conf.append(np.zeros((J,), dtype=np.float32))
            continue

        res = pose.process(rgb)

        if res.pose_landmarks and res.pose_landmarks.landmark:
            lm = res.pose_landmarks.landmark

            # xy are normalized to [0,1] relative to image width/height.
            xy = np.array([[lm[i].x, lm[i].y] for i in range(J)], dtype=np.float32)

            # visibility is a MediaPipe-provided confidence/visibility score.
            conf = np.array([lm[i].visibility for i in range(J)], dtype=np.float32)
        else:
            # No pose detected in this frame -> fill zeros.
            xy = np.zeros((J, 2), dtype=np.float32)
            conf = np.zeros((J,), dtype=np.float32)

        frames_xy.append(xy)
        frames_conf.append(conf)

        # Progress logging (time-based to avoid spamming)
        now = time.time()
        if log_every_s > 0 and (now - last_log) >= log_every_s:
            print(f"[pose] {os.path.basename(path)} : {frame_idx} frames...", flush=True)
            last_log = now

    cap.release()

    if not frames_xy:
        print(f"[warn] no frames read: {path}", file=sys.stderr)
        return False

    xy = np.stack(frames_xy, axis=0)      # [T,33,2]
    conf = np.stack(frames_conf, axis=0)  # [T,33]

    payload = dict(
        xy=xy.astype(np.float32),
        conf=conf.astype(np.float32),
        fps=np.float32(fps),
        size=np.array([w, h], dtype=np.int32),
        src=str(path),
    )
    _atomic_save_npz(out_npz, payload)

    print(f"[ok] {path} → {out_npz}  (T={xy.shape[0]}, fps={fps:.1f}, size={w}x{h})")
    return True


# ============================================================
# 5) Main
# ============================================================
def main() -> None:
    args = parse_args()

    videos = list_videos(args.videos_glob)
    if args.max_videos:
        videos = videos[: max(0, int(args.max_videos))]

    if not videos:
        raise SystemExit(f"[ERR] No videos matched: {args.videos_glob}")

    root = common_root(videos)
    os.makedirs(args.out_dir, exist_ok=True)

    mp_pose = mp.solutions.pose

    # Create a single Pose object and reuse it for all videos.
    # This is faster than recreating it per video.
    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as pose:

        ok_count = 0
        for vp in videos:
            stem = make_safe_stem(vp, root)
            out = os.path.join(args.out_dir, f"{stem}.npz")

            if args.skip_existing and os.path.exists(out):
                print(f"[skip] exists: {out}")
                ok_count += 1
                continue

            try:
                ok = run_one_video(
                    vp,
                    out,
                    pose,
                    fps_default=args.fps_default,
                    force_fps=args.force_fps,
                    log_every_s=args.log_every_s,
                )
                ok_count += int(ok)
            except KeyboardInterrupt:
                print("\n[abort] interrupted by user", file=sys.stderr)
                break
            except Exception as e:
                print(f"[err] {vp}: {e}", file=sys.stderr)

    print(f"[done] processed {ok_count}/{len(videos)} videos → {args.out_dir}")


if __name__ == "__main__":
    main()
