#!/usr/bin/env python3
"""
extract_2d.py

Extract 2D human pose from videos using MediaPipe Pose and save as NPZ.

Per-video NPZ contains:
  - xy:   float32 [T, 33, 2]   normalized coords (0..1)
  - conf: float32 [T, 33]      visibility/confidence
  - fps:  float32              effective pose FPS (after optional frame skipping)
  - src_fps: float32           original source FPS from video metadata
  - frame_step: int32          processing step (1=no skipping)
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


def _write_landmarks_xy_conf(lm, xy_out: np.ndarray, conf_out: np.ndarray) -> None:
    # Fixed-size path with upfront zero fill keeps missing semantics consistent
    # and avoids per-index bounds branches on every frame.
    isfinite = np.isfinite
    xy_out.fill(0.0)
    conf_out.fill(0.0)
    n_lm = min(J, int(len(lm)) if lm is not None else 0)
    for i in range(n_lm):
        p = lm[i]
        x = float(p.x)
        y = float(p.y)
        v = float(p.visibility)
        if not (isfinite(x) and isfinite(y) and isfinite(v)):
            continue
        if x <= 0.0:
            xy_out[i, 0] = 0.0
        elif x >= 1.0:
            xy_out[i, 0] = 1.0
        else:
            xy_out[i, 0] = x
        if y <= 0.0:
            xy_out[i, 1] = 0.0
        elif y >= 1.0:
            xy_out[i, 1] = 1.0
        else:
            xy_out[i, 1] = y
        if v <= 0.0:
            conf_out[i] = 0.0
        elif v >= 1.0:
            conf_out[i] = 1.0
        else:
            conf_out[i] = v


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
    ap.add_argument("--max_proc_fps", type=float, default=0.0,
                    help="Optional cap for pose processing FPS. "
                         "If >0 and source FPS is higher, process every Nth frame.")
    return ap.parse_args()


def list_videos(patterns: Iterable[str]) -> List[str]:
    """Expand globs inside Python, deduplicate, and sort deterministically."""
    files: set[str] = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if os.path.isfile(f):
                files.add(f)
    return sorted(files)


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
    max_proc_fps: float = 0.0,
) -> bool:
    """Read video frame-by-frame, run MediaPipe Pose, collect arrays, save NPZ."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] cannot open: {path}", file=sys.stderr)
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if force_fps is not None:
        fps = float(force_fps)
    elif (not fps) or (not np.isfinite(fps)) or fps <= 0.0:
        fps = float(fps_default)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    size_arr = np.array([w, h], dtype=np.int32)

    frame_idx = 0
    write_idx = 0
    last_log = time.time()
    base_name = os.path.basename(path)
    pose_process = pose.process
    cvt_color = cv2.cvtColor
    bgr2rgb = cv2.COLOR_BGR2RGB
    zero_xy = np.zeros((J, 2), dtype=np.float32)
    zero_conf = np.zeros((J,), dtype=np.float32)
    proc_fps = float(max_proc_fps) if float(max_proc_fps) > 0.0 else 0.0
    frame_step = 1
    if proc_fps > 0.0 and fps > proc_fps:
        frame_step = max(1, int(round(fps / proc_fps)))
    src_fps = float(fps)
    fps_eff = float(fps / frame_step) if frame_step > 1 else float(fps)
    if frame_step > 1:
        print(f"[pose] downsample {base_name}: src_fps={src_fps:.2f}, step={frame_step}, proc_fps~{fps_eff:.2f}")

    frame_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_est > 0 and frame_step > 1:
        # Preallocate by expected processed-frame count, not full source-frame count.
        frame_est = int((frame_est + frame_step - 1) // frame_step)
    use_prealloc = frame_est > 0
    if use_prealloc:
        xy_main = np.zeros((frame_est, J, 2), dtype=np.float32)
        conf_main = np.zeros((frame_est, J), dtype=np.float32)
        overflow_xy: List[np.ndarray] = []
        overflow_conf: List[np.ndarray] = []
    else:
        frames_xy: List[np.ndarray] = []
        frames_conf: List[np.ndarray] = []

    def push_frame(xy_fr: Optional[np.ndarray], conf_fr: Optional[np.ndarray]) -> None:
        nonlocal write_idx
        if use_prealloc and write_idx < frame_est:
            # Preallocated buffers are already zero-initialized; skip writes for missing frames.
            if xy_fr is not None and conf_fr is not None:
                xy_main[write_idx] = xy_fr
                conf_main[write_idx] = conf_fr
        elif use_prealloc:
            overflow_xy.append(zero_xy if xy_fr is None else xy_fr)
            overflow_conf.append(zero_conf if conf_fr is None else conf_fr)
        else:
            frames_xy.append(zero_xy if xy_fr is None else xy_fr)
            frames_conf.append(zero_conf if conf_fr is None else conf_fr)
        write_idx += 1

    try:
        while True:
            # Use grab/retrieve for frame skipping to avoid decoding frames that
            # will be dropped by the max processing FPS cap.
            if frame_step > 1:
                ok = cap.grab()
                if not ok:
                    break
                frame_idx += 1
                if ((frame_idx - 1) % frame_step) != 0:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

            # If container metadata is missing, recover width/height from first valid frame.
            if (w == 0 or h == 0) and frame is not None and hasattr(frame, "shape"):
                fh, fw = frame.shape[:2]
                if fw > 0 and fh > 0:
                    w = fw if w == 0 else w
                    h = fh if h == 0 else h
                    size_arr[0] = w
                    size_arr[1] = h

            # Some decoders can return ok=True but frame=None; handle safely.
            if frame is None:
                push_frame(None, None)
                continue

            # BGR -> RGB (MediaPipe expects RGB)
            try:
                rgb = cvt_color(frame, bgr2rgb)
            except Exception:
                push_frame(None, None)
                continue
            rgb.flags.writeable = False

            res = pose_process(rgb)

            if res.pose_landmarks and res.pose_landmarks.landmark:
                lm = res.pose_landmarks.landmark
                if use_prealloc and write_idx < frame_est:
                    _write_landmarks_xy_conf(lm, xy_main[write_idx], conf_main[write_idx])
                    write_idx += 1
                else:
                    xy = np.empty((J, 2), dtype=np.float32)
                    conf = np.empty((J,), dtype=np.float32)
                    _write_landmarks_xy_conf(lm, xy, conf)
                    push_frame(xy, conf)
            else:
                push_frame(None, None)

            now = time.time()
            if log_every_s > 0 and (now - last_log) >= log_every_s:
                print(f"[pose] {base_name} : {frame_idx} frames...", flush=True)
                last_log = now

    finally:
        cap.release()

    if write_idx < 1:
        print(f"[warn] no frames read: {path}", file=sys.stderr)
        return False

    if use_prealloc:
        xy = xy_main[:write_idx]
        conf = conf_main[:write_idx]
        if overflow_xy:
            xy = np.concatenate([xy, np.stack(overflow_xy, axis=0)], axis=0)
            conf = np.concatenate([conf, np.stack(overflow_conf, axis=0)], axis=0)
    else:
        xy = np.stack(frames_xy, axis=0)         # [T,33,2]
        conf = np.stack(frames_conf, axis=0)     # [T,33]

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    # Atomic write: avoid partially-written files being skipped on rerun.
    tmp_npz = out_npz + ".tmp.npz"
    np.savez_compressed(
        tmp_npz,
        xy=xy.astype(np.float32, copy=False),
        conf=conf.astype(np.float32, copy=False),
        fps=np.float32(fps_eff),
        src_fps=np.float32(src_fps),
        frame_step=np.int32(frame_step),
        size=size_arr,
        src=str(path),
    )
    os.replace(tmp_npz, out_npz)
    print(f"[ok] {path} → {out_npz}  (T={xy.shape[0]}, fps={fps_eff:.1f}, size={w}x{h})")
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
                    max_proc_fps=args.max_proc_fps,
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
