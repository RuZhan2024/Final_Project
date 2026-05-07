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

# Silence verbose logs BEFORE importing mediapipe backends (best effort)
os.environ.setdefault("GLOG_minloglevel", "2")     # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Force CPU-safe graph startup in headless/CI/macOS shells where NSOpenGL may be unavailable.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import numpy as np
import cv2
import mediapipe as mp

J = 33  # MediaPipe Pose landmarks
_DIGIT_SPLIT_RE = re.compile(r"(\d+)")

DEFAULT_FPS = {
    "urfd": 30.0,
    "caucafall": 23.0,
    "muvim": 30.0,
}

def _natural_key(s: str):
    """Natural sort key: frame_2.png < frame_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in _DIGIT_SPLIT_RE.split(s)]


def _sanitize_stem(stem: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


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


def _write_landmarks_xy_conf_mapped(
    lm,
    xy_out: np.ndarray,
    conf_out: np.ndarray,
    *,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    full_w: int,
    full_h: int,
) -> None:
    isfinite = np.isfinite
    xy_out.fill(0.0)
    conf_out.fill(0.0)
    n_lm = min(J, int(len(lm)) if lm is not None else 0)
    if crop_w <= 0 or crop_h <= 0 or full_w <= 0 or full_h <= 0:
        return
    for i in range(n_lm):
        p = lm[i]
        x = float(p.x)
        y = float(p.y)
        v = float(p.visibility)
        if not (isfinite(x) and isfinite(y) and isfinite(v)):
            continue
        xf = (float(x0) + x * float(crop_w)) / float(full_w)
        yf = (float(y0) + y * float(crop_h)) / float(full_h)
        xy_out[i, 0] = float(np.clip(xf, 0.0, 1.0))
        xy_out[i, 1] = float(np.clip(yf, 0.0, 1.0))
        conf_out[i] = float(np.clip(v, 0.0, 1.0))


def _read_yolo_bbox(frame_path: str):
    txt_path = os.path.splitext(frame_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                if bw > 0.0 and bh > 0.0:
                    return xc, yc, bw, bh
    except Exception:
        return None
    return None


def _bbox_crop_rect(frame_path: str, width: int, height: int, scale: float):
    bbox = _read_yolo_bbox(frame_path)
    if bbox is None or width <= 0 or height <= 0:
        return None
    xc, yc, bw, bh = bbox
    scale = max(1.0, float(scale))
    cx = xc * width
    cy = yc * height
    crop_w = bw * width * scale
    crop_h = bh * height * scale
    x0 = int(np.floor(cx - crop_w / 2.0))
    y0 = int(np.floor(cy - crop_h / 2.0))
    x1 = int(np.ceil(cx + crop_w / 2.0))
    y1 = int(np.ceil(cy + crop_h / 2.0))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None
    return x0, y0, x1, y1


def list_sequences(image_globs, sequence_id_depth: int):
    """
    Return:
      frames_by_seq: dict[seq_id, list[frame_path]]
      all_paths: list[str]
    """
    frames_by_seq = defaultdict(list)
    all_paths_set = set()

    globs_list = image_globs if isinstance(image_globs, list) else [image_globs]

    for g in globs_list:
        for p in glob.glob(g, recursive=True):
            if os.path.isfile(p):
                all_paths_set.add(p)

    all_paths = sorted(all_paths_set)

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

    # Sort frame order within each sequence only (natural order by filename).
    for seq_id, frames in frames_by_seq.items():
        frames.sort(key=lambda p: _natural_key(os.path.basename(p)))

    # Stable sequence iteration order for deterministic output.
    seq_sorted = {k: frames_by_seq[k] for k in sorted(frames_by_seq.keys())}
    return seq_sorted, all_paths


def _safe_out_name(seq_id: str, src_rel: str) -> str:
    """
    Stable filename from seq_id + short hash of *relative* source directory.

    Using a relative path keeps filenames reproducible across machines.
    """
    base = _sanitize_stem(seq_id.replace("/", "__"))
    src_rel_norm = src_rel.replace(os.sep, "/")
    h = hashlib.md5(src_rel_norm.encode("utf-8")).hexdigest()[:8]
    return f"{base}__{h}.npz"


def extract_sequence(
    frames,
    pose,
    out_npz: str,
    fps: float,
    src: str,
    seq_id: str,
    store_frames: bool = False,
    bbox_crop_mode: str = "off",
    bbox_crop_scale: float = 1.25,
):
    """
    Save:
      xy  [T,33,2] (0 when missing)
      conf[T,33]
      fps, size, src, seq_id
    """
    n_frames = len(frames)
    xy = np.zeros((n_frames, J, 2), dtype=np.float32)
    cf = np.zeros((n_frames, J), dtype=np.float32)
    size = np.array([0, 0], dtype=np.int32)  # [w,h]
    imread = cv2.imread
    cvt_color = cv2.cvtColor
    bgr2rgb = cv2.COLOR_BGR2RGB
    pose_process = pose.process

    for i, f in enumerate(frames):
        img = imread(f)
        if img is None:
            continue

        if size[0] == 0 and size[1] == 0:
            h, w = img.shape[:2]
            size[0] = w
            size[1] = h

        h, w = img.shape[:2]
        crop_rect = None
        if bbox_crop_mode in {"fallback", "always"}:
            crop_rect = _bbox_crop_rect(f, w, h, bbox_crop_scale)

        if bbox_crop_mode == "always" and crop_rect is not None:
            x0, y0, x1, y1 = crop_rect
            rgb = cvt_color(img[y0:y1, x0:x1], bgr2rgb)
            rgb.flags.writeable = False
            res = pose_process(rgb)
            if res.pose_landmarks and res.pose_landmarks.landmark:
                lm = res.pose_landmarks.landmark
                _write_landmarks_xy_conf_mapped(
                    lm,
                    xy[i],
                    cf[i],
                    x0=x0,
                    y0=y0,
                    crop_w=x1 - x0,
                    crop_h=y1 - y0,
                    full_w=w,
                    full_h=h,
                )
                continue

        rgb = cvt_color(img, bgr2rgb)
        rgb.flags.writeable = False
        res = pose_process(rgb)

        if res.pose_landmarks and res.pose_landmarks.landmark:
            lm = res.pose_landmarks.landmark
            _write_landmarks_xy_conf(lm, xy[i], cf[i])
            continue

        if bbox_crop_mode == "fallback" and crop_rect is not None:
            x0, y0, x1, y1 = crop_rect
            rgb_crop = cvt_color(img[y0:y1, x0:x1], bgr2rgb)
            rgb_crop.flags.writeable = False
            res_crop = pose_process(rgb_crop)
            if res_crop.pose_landmarks and res_crop.pose_landmarks.landmark:
                lm = res_crop.pose_landmarks.landmark
                _write_landmarks_xy_conf_mapped(
                    lm,
                    xy[i],
                    cf[i],
                    x0=x0,
                    y0=y0,
                    crop_w=x1 - x0,
                    crop_h=y1 - y0,
                    full_w=w,
                    full_h=h,
                )

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        xy=xy.astype(np.float32, copy=False),
        conf=cf.astype(np.float32, copy=False),
        fps=np.float32(fps),
        size=size,
        src=np.str_(src),
        seq_id=np.str_(seq_id),
    )
    if store_frames:
        max_len = max((len(str(p)) for p in frames), default=1)
        payload["frames"] = np.asarray(frames, dtype=f"<U{max_len}")

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
    ap.add_argument("--source_root", default=None,
                    help="Optional stable source root used for output filename hashes.")
    ap.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2],
                    help="0=lite,1=full,2=heavy (default).")
    ap.add_argument("--min_det_conf", type=float, default=0.5,
                    help="Min detection confidence.")
    ap.add_argument("--min_track_conf", type=float, default=0.5,
                    help="Min tracking confidence.")
    ap.add_argument("--static_image_mode", action="store_true",
                    help="If set, disables temporal tracking (usually slower for sequences).")
    ap.add_argument("--bbox_crop_mode", choices=["off", "fallback", "always"], default="off",
                    help="Use sidecar YOLO txt boxes to crop before MediaPipe. fallback keeps full-frame first.")
    ap.add_argument("--bbox_crop_scale", type=float, default=1.25,
                    help="Expansion factor for sidecar bbox crops.")
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
    if args.source_root:
        common_root = os.path.abspath(args.source_root)
    else:
        try:
            common_root = os.path.commonpath(all_paths)
        except Exception:
            common_root = os.path.dirname(all_paths[0])

    seq_items = list(frames_by_seq.items())
    if args.max_sequences is not None:
        seq_items = seq_items[: max(0, int(args.max_sequences))]

    print(f"[INFO] Found {len(frames_by_seq)} sequences, using {len(seq_items)} sequences")

    # Precompute per-sequence source dir + relative source for stable naming/logging.
    seq_src_rel: dict[str, str] = {}
    for seq_id, frames in seq_items:
        if not frames:
            seq_src_rel[seq_id] = ""
            continue
        try:
            seq_src_dir = os.path.commonpath(frames)
        except Exception:
            seq_src_dir = os.path.dirname(frames[0])
        try:
            src_rel = os.path.relpath(seq_src_dir, common_root)
        except Exception:
            src_rel = seq_src_dir
        seq_src_rel[seq_id] = src_rel

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    ) as pose:

        for i, (seq_id, frames) in enumerate(seq_items, start=1):
            src_rel = seq_src_rel.get(seq_id, "")

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
                bbox_crop_mode=args.bbox_crop_mode,
                bbox_crop_scale=float(args.bbox_crop_scale),
            )

    print(f"[OK] wrote {len(seq_items)} sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
