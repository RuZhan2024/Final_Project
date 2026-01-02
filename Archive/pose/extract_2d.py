#!/usr/bin/env python3
"""
extract_2d.py

Extract 2D human pose from videos using MediaPipe Pose and save as NPZ.

Per-video NPZ contains:
  - xy:   float32 [T, 33, 2]   normalized coords (0..1)
  - conf: float32 [T, 33]      visibility/confidence
  - fps:  float32              source FPS (fallback if unknown)
  - size: int32   [2]          [width, height] in pixels
  - src:  str                  original video path

Usage example:
  python pose/extract_2d.py \
    --videos_glob 'data/raw/LE2i/**/videos/*.mp4' 'data/raw/LE2i/**/*.avi' \
    --out_dir 'data/interim/le2i/pose_npz'
"""

# Silence verbose logs BEFORE importing mediapipe/tf backends (best effort)
import os as _os
_os.environ.setdefault("GLOG_minloglevel", "2")     # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import cv2
import mediapipe as mp

J = 33  # MediaPipe Pose landmarks


def parse_args():
    ap = argparse.ArgumentParser(description="Extract 2D pose (MediaPipe) from videos into per-video NPZs.")
    ap.add_argument("--videos_glob", nargs="+", required=True,
                    help="One or more quoted glob patterns. Example: '.../*.mp4' '.../*.avi'")
    ap.add_argument("--out_dir", required=True, help="Directory to write NPZ files.")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                    help="0=lite, 1=full (default), 2=heavy")
    ap.add_argument("--min_det_conf", type=float, default=0.5, help="Min detection confidence.")
    ap.add_argument("--min_track_conf", type=float, default=0.5, help="Min tracking confidence.")
    ap.add_argument("--static_image_mode", action="store_true",
                    help="If set, disables temporal tracking (slower).")
    ap.add_argument("--max_videos", type=int, default=None, help="Optional cap for debugging (process only first N).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip processing if output NPZ already exists.")
    ap.add_argument("--fps_default", type=float, default=30.0,
                    help="Fallback FPS if video metadata FPS is missing/0 (default: 30).")
    ap.add_argument("--force_fps", type=float, default=None,
                    help="Override and store this FPS for all videos (useful for dataset-level correction).")
    ap.add_argument("--log_every_s", type=float, default=3.0,
                    help="Print progress at most once every N seconds per video (default 3s).")
    return ap.parse_args()


def list_videos(patterns: Iterable[str]) -> List[str]:
    """Expand globs inside Python, deduplicate, and sort deterministically."""
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return sorted(set(files))


def common_root(paths: List[str]) -> str:
    if not paths:
        return "."
    if len(paths) == 1:
        return os.path.dirname(paths[0])
    try:
        return os.path.commonpath(paths)
    except ValueError:
        return os.path.dirname(paths[0])


def make_safe_stem(video_path: str, root: str) -> str:
    """
    Build a collision-free NPZ filename by encoding the relative path:
      'Coffee_room_01/videos/fall01.mp4' → 'Coffee_room_01__videos__fall01'
    """
    rel = os.path.relpath(video_path, root)
    stem = os.path.splitext(rel)[0].replace(os.sep, "__")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


def run_one_video(
    path: str,
    out_npz: str,
    pose,
    fps_default: float = 30.0,
    force_fps: Optional[float] = None,
    log_every_s: float = 3.0,
) -> bool:
    """Read video frame-by-frame, run MediaPipe Pose, collect arrays, save NPZ."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] cannot open: {path}", file=sys.stderr)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if force_fps is not None:
        fps = float(force_fps)
    elif (not fps) or np.isnan(fps) or fps <= 0.0:
        fps = float(fps_default)

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

        # BGR -> RGB (MediaPipe expects RGB)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frames_xy.append(np.zeros((J, 2), dtype=np.float32))
            frames_conf.append(np.zeros((J,), dtype=np.float32))
            continue

        res = pose.process(rgb)

        if res.pose_landmarks and res.pose_landmarks.landmark:
            lm = res.pose_landmarks.landmark
            xy = np.array([[lm[i].x, lm[i].y] for i in range(J)], dtype=np.float32)  # [33,2]
            conf = np.array([lm[i].visibility for i in range(J)], dtype=np.float32)  # [33]
        else:
            xy = np.zeros((J, 2), dtype=np.float32)
            conf = np.zeros((J,), dtype=np.float32)

        frames_xy.append(xy)
        frames_conf.append(conf)

        now = time.time()
        if log_every_s > 0 and (now - last_log) >= log_every_s:
            print(f"[pose] {os.path.basename(path)} : {frame_idx} frames...", flush=True)
            last_log = now

    cap.release()

    if not frames_xy:
        print(f"[warn] no frames read: {path}", file=sys.stderr)
        return False

    xy = np.stack(frames_xy, axis=0)         # [T,33,2]
    conf = np.stack(frames_conf, axis=0)     # [T,33]

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        xy=xy.astype(np.float32),
        conf=conf.astype(np.float32),
        fps=np.float32(fps),
        size=np.array([w, h], dtype=np.int32),
        src=str(path),
    )
    print(f"[ok] {path} → {out_npz}  (T={xy.shape[0]}, fps={fps:.1f}, size={w}x{h})")
    return True


def main():
    args = parse_args()
    videos = list_videos(args.videos_glob)
    if args.max_videos:
        videos = videos[: max(0, int(args.max_videos))]

    if not videos:
        raise SystemExit(f"[ERR] No videos matched: {args.videos_glob}")

    root = common_root(videos)
    os.makedirs(args.out_dir, exist_ok=True)

    mp_pose = mp.solutions.pose
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
