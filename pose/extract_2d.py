#!/usr/bin/env python3
"""
Extract 2D human pose from videos using MediaPipe Pose and save as NPZ.

Per-video NPZ contains:
  - xy:   float32 [T, 33, 2]   normalized coords (0..1)
  - conf: float32 [T, 33]      visibility/confidence
  - fps:  float32              source FPS (fallback 30 if unknown)
  - size: int32   [2]          [width, height] in pixels
  - src:  str                  original video path

Usage example:
  python pose/extract_2d.py \
    --videos_glob 'data/raw/LE2i/**/videos/*.mp4' 'data/raw/LE2i/**/*.avi' \
    --out_dir 'data/interim/le2i/pose_npz'
"""

# Silence verbose logs BEFORE importing mediapipe/tf backends
import os as _os
_os.environ.setdefault("GLOG_minloglevel", "2")     # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import os, sys, glob, argparse, time, pathlib
import numpy as np
import cv2
import mediapipe as mp

J = 33  # number of MediaPipe Pose landmarks

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_glob", nargs="+", required=True,
                    help="One or more quoted glob patterns. Example: '.../*.mp4' '.../*.avi'")
    ap.add_argument("--out_dir", required=True, help="Directory to write NPZ files.")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                    help="0=lite, 1=full (default), 2=heavy")
    ap.add_argument("--min_det_conf", type=float, default=0.5,
                    help="Min detection confidence.")
    ap.add_argument("--min_track_conf", type=float, default=0.5,
                    help="Min tracking confidence.")
    ap.add_argument("--static_image_mode", action="store_true",
                    help="If set, disables temporal tracking (slower).")
    ap.add_argument("--max_videos", type=int, default=None,
                    help="Optional cap for debugging (process only first N).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip processing if output NPZ already exists.")
    return ap.parse_args()

def list_videos(patterns):
    """Expand one or more globs INSIDE Python and deduplicate/sort."""
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted({f for f in files if os.path.isfile(f)})
    return files

def make_safe_stem(video_path, common_root):
    """
    Build a collision-free NPZ filename by encoding the relative path:
      'Coffee_room_01/videos/fall01.mp4' → 'Coffee_room_01__videos__fall01'
    """
    rel = os.path.relpath(video_path, common_root)
    stem = os.path.splitext(rel)[0].replace(os.sep, "__")
    # clean any funky characters
    stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)
    return stem

def run_one_video(path, out_npz, pose):
    """Read video frame-by-frame, run MediaPipe Pose, collect arrays, save NPZ."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[skip] cannot open: {path}", file=sys.stderr)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or np.isnan(fps):
        fps = 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_xy, frames_conf = [], []
    frame_idx, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # BGR -> RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # In static_image_mode=False, tracking makes it faster after first detection
        res = pose.process(rgb)

        if res.pose_landmarks and res.pose_landmarks.landmark:
            lm = res.pose_landmarks.landmark
            # landmarks are normalized [0,1] by image size (x,y)
            xy = np.array([[lm[i].x, lm[i].y] for i in range(J)], dtype=np.float32)  # [33,2]
            conf = np.array([lm[i].visibility for i in range(J)], dtype=np.float32)  # [33]
        else:
            # No detection → fill zeros (simple for downstream)
            xy = np.zeros((J, 2), dtype=np.float32)
            conf = np.zeros((J,), dtype=np.float32)

        frames_xy.append(xy)
        frames_conf.append(conf)

        # tiny progress log every ~3s
        if frame_idx % 120 == 0 and (time.time() - t0) > 3:
            print(f"[pose] {os.path.basename(path)} : {frame_idx} frames...", flush=True)
            t0 = time.time()

    cap.release()

    if not frames_xy:
        print(f"[warn] no frames read: {path}", file=sys.stderr)
        return False

    xy   = np.stack(frames_xy,  axis=0)       # [T,33,2]
    conf = np.stack(frames_conf, axis=0)      # [T,33]

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
        videos = videos[:args.max_videos]
    if not videos:
        sys.exit(f"No videos matched: {args.videos_glob}")

    # common root for readable, collision-free stems
    try:
        common_root = os.path.commonpath(videos) if len(videos) > 1 else os.path.dirname(videos[0])
    except ValueError:
        # fallback if paths are on different drives (unlikely on macOS)
        common_root = os.path.dirname(videos[0])

    os.makedirs(args.out_dir, exist_ok=True)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=args.static_image_mode,
                      model_complexity=args.model_complexity,
                      enable_segmentation=False,
                      min_detection_confidence=args.min_det_conf,
                      min_tracking_confidence=args.min_track_conf) as pose:

        ok_count = 0
        for vp in videos:
            stem = make_safe_stem(vp, common_root)
            out = os.path.join(args.out_dir, f"{stem}.npz")

            if args.skip_existing and os.path.exists(out):
                print(f"[skip] exists: {out}")
                ok_count += 1
                continue

            try:
                ok = run_one_video(vp, out, pose)
                ok_count += int(ok)
            except KeyboardInterrupt:
                print("\n[abort] interrupted by user", file=sys.stderr)
                break
            except Exception as e:
                print(f"[err] {vp}: {e}", file=sys.stderr)

    print(f"[done] processed {ok_count}/{len(videos)} videos → {args.out_dir}")

if __name__ == "__main__":
    main()
