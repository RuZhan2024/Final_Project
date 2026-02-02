#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_urfd_labels.py

Build UR_Fall sequence-level labels + fall spans for the skeleton pipeline.

UR_Fall raw format (Roboflow YOLO export)
-----------------------------------------
- raw_root/{train,valid,test}/...*.txt  (YOLO per-frame labels)
- Optional raw_root/data.yaml defines class names.

Your pipeline needs:
- labels keyed by pose sequence stem in npz_dir
- spans in sequence-index space [start, end_excl)

This script:
1) Reads pose stems from --npz_dir (canonical keys).
2) Extracts the clip key (e.g., 'fall-03-cam1-rgb') from each stem.
3) Scans raw YOLO labels to build a boolean "fall frame" flag list per clip.
4) Applies optional gap-fill to smooth noisy detections.
5) Extracts fall runs (spans).
6) Optionally maps raw frame index -> pose index via --frame_stride.
7) Optionally clamps spans to NPZ length.

Outputs:
- out_labels JSON: {stem: 0/1}
- out_spans  JSON: {stem: [[start, end_excl], ...]}  (only for positives)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Match clip identifiers present in UR_Fall exports and in your sequence stems
CLIP_RE = re.compile(r"(?:adl|fall)-\d+-cam\d+-rgb", re.IGNORECASE)


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _parse_frame_idx_from_stem(stem: str) -> Optional[int]:
    """
    Raw frame label stems often look like:
      adl-01-cam0-rgb-091_png
    or:
      fall-03-cam1-rgb-012_jpg

    We extract the integer right before '_png/_jpg/_jpeg'.
    """
    m = re.search(r"-(\d+)_(?:png|jpg|jpeg)$", stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _clipkey_from_frame_stem(stem: str) -> str:
    """Remove trailing '-NNN_png' and normalize to lowercase."""
    return re.sub(r"-\d+_(?:png|jpg|jpeg)$", "", stem, flags=re.IGNORECASE).lower()


def _clipkey_from_seq_stem(seq_stem: str) -> Optional[str]:
    """
    Pose stem formats seen:
      - adl__adl-02-cam0-rgb__bf15e056
      - adl-02-cam0-rgb__6a478ea8

    We regex-search for the clip key.
    """
    m = CLIP_RE.search(seq_stem)
    return m.group(0).lower() if m else None


def _is_fall_frame(txt_path: str, fall_class_id: int) -> bool:
    """
    Frame is considered 'fall' if ANY line in YOLO txt has class id == fall_class_id.

    YOLO line format:
      <class> <x> <y> <w> <h>
    We only need the first integer.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # parse the first token as int class id
                tok = line.split()[0]
                try:
                    cid = int(tok)
                except Exception:
                    continue
                if cid == fall_class_id:
                    return True
    except FileNotFoundError:
        return False
    return False


def _gap_fill(flags: List[bool], gap_fill: int) -> List[bool]:
    """
    Fill short False gaps inside True segments.

    Example:
      True True False True True  with gap_fill>=1 becomes all True.
    """
    if gap_fill <= 0 or not flags:
        return flags
    flags = flags[:]  # copy
    i = 0
    while i < len(flags):
        if flags[i]:
            i += 1
            continue
        j = i
        while j < len(flags) and not flags[j]:
            j += 1
        gap = j - i
        if i > 0 and j < len(flags) and flags[i - 1] and flags[j] and gap <= gap_fill:
            for t in range(i, j):
                flags[t] = True
        i = j
    return flags


def _runs_from_flags(flags: List[bool], min_run: int) -> List[List[int]]:
    """
    Convert a boolean list to contiguous True runs:
      flags: [False, True, True, False] -> [[1, 3]]

    Runs are returned as [start, end_excl].
    """
    runs: List[List[int]] = []
    start: Optional[int] = None
    for t, b in enumerate(flags):
        if b and start is None:
            start = t
        elif (not b) and start is not None:
            end = t
            if end - start >= min_run:
                runs.append([start, end])
            start = None
    if start is not None:
        end = len(flags)
        if end - start >= min_run:
            runs.append([start, end])
    return runs


def _infer_fall_class_id_from_data_yaml(raw_root: str) -> Optional[int]:
    """
    Try to infer fall class id from data.yaml if present.
    Expected YAML structure:
      names: ['fall', 'person', ...]
    Class id is the index in the list.
    """
    path = os.path.join(raw_root, "data.yaml")
    if not os.path.exists(path):
        return None
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        if isinstance(names, list):
            for i, n in enumerate(names):
                if str(n).strip().lower() == "fall":
                    return int(i)
    except Exception:
        return None
    return None


def seq_len_from_npz(npz_path: str) -> Optional[int]:
    """Read length from pose NPZ (xy or joints)."""
    try:
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
            if "joints" in z:
                return int(z["joints"].shape[0])
    except Exception:
        return None
    return None


def map_runs_with_stride_and_clamp(
    runs: List[List[int]],
    *,
    frame_stride: int,
    clamp_len: Optional[int],
) -> List[List[int]]:
    """
    Map raw-frame runs -> pose-frame runs.

    For end-exclusive spans [s, e_excl):
      s_pose = floor(s / k)
      e_pose = ceil(e_excl / k) = (e_excl + k - 1)//k
    """
    k = max(1, int(frame_stride))
    out: List[List[int]] = []
    for s, e in runs:
        s_pose = int(s) // k
        e_pose = (int(e) + k - 1) // k
        if clamp_len is not None:
            s_pose = max(0, min(s_pose, clamp_len))
            e_pose = max(0, min(e_pose, clamp_len))
        if e_pose > s_pose:
            out.append([s_pose, e_pose])

    # merge overlaps just in case
    out.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Roboflow UR_Fall root (contains train/valid/test)")
    ap.add_argument("--npz_dir", required=True, help="Pose sequence NPZ dir (canonical stems). NOT windows dir.")
    ap.add_argument("--out_labels", default="configs/labels/urfd.json")
    ap.add_argument("--out_spans", default="configs/labels/urfd_spans.json")

    ap.add_argument("--min_run", type=int, default=2, help="Minimum run length (frames) for a fall span.")
    ap.add_argument("--gap_fill", type=int, default=2, help="Fill short detection gaps up to this many frames.")
    ap.add_argument(
        "--fall_class_id",
        type=int,
        default=-1,
        help="Override fall class id (0-based). If -1, infer from data.yaml or fallback to 0.",
    )

    ap.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="If pose extraction used every k-th raw frame, set k here (default: 1).",
    )
    ap.add_argument(
        "--clamp_to_npz_len",
        action="store_true",
        help="Clamp spans to pose NPZ length (best-effort).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    raw_root = args.raw_root
    npz_dir = args.npz_dir

    # Determine fall class id
    fall_id = int(args.fall_class_id)
    if fall_id < 0:
        fall_id = _infer_fall_class_id_from_data_yaml(raw_root) or 0

    # 1) Pose stems (canonical keys)
    pose_npzs = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in pose_npzs]
    if not stems:
        raise SystemExit(f"[ERR] No pose NPZ files found in {npz_dir}")

    if args.verbose:
        print(f"[info] pose_npz files: {len(stems)}  dir={npz_dir}")
        print(f"[info] fall_class_id={fall_id}")

    # 2) Build clipkey -> ordered list of frame label paths from raw YOLO labels
    clip_frames: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for split in ["train", "valid", "test", "val"]:
        d = os.path.join(raw_root, split)
        if not os.path.isdir(d):
            continue

        # Roboflow sometimes puts txt directly in split dir, sometimes in split/labels/
        patterns = [os.path.join(d, "*.txt"), os.path.join(d, "labels", "*.txt")]
        for pat in patterns:
            for p in glob.glob(pat):
                stem = os.path.splitext(os.path.basename(p))[0]
                idx = _parse_frame_idx_from_stem(stem)
                if idx is None:
                    continue
                ck = _clipkey_from_frame_stem(stem)
                clip_frames[ck].append((idx, p))

    # Sort each clip by frame index
    for ck in list(clip_frames.keys()):
        clip_frames[ck].sort(key=lambda x: x[0])

    if args.verbose:
        print(f"[info] raw clipkeys discovered: {len(clip_frames)}")

    # 3) Compute fall runs per clipkey (still in raw-frame index space)
    clip_runs_raw: Dict[str, List[List[int]]] = {}
    for ck, items in clip_frames.items():
        flags = [_is_fall_frame(p, fall_id) for _, p in items]
        flags = _gap_fill(flags, int(args.gap_fill))
        clip_runs_raw[ck] = _runs_from_flags(flags, int(args.min_run))

    # 4) Emit labels/spans keyed by pose stem
    labels: Dict[str, int] = {}
    spans: Dict[str, List[List[int]]] = {}
    missing_clipkey = 0
    fall_total = 0

    for npz_path, stem in zip(pose_npzs, stems):
        ck = _clipkey_from_seq_stem(stem)
        if ck is None:
            missing_clipkey += 1
            labels[stem] = 1 if "fall-" in stem.lower() else 0
            continue

        # URFD naming: 'fall-*' is positive, 'adl-*' is negative
        y = 1 if ck.startswith("fall-") else 0
        labels[stem] = y

        if y == 1:
            fall_total += 1
            runs_raw = clip_runs_raw.get(ck, [])

            # Map to pose indices + optional clamp
            clamp_len = seq_len_from_npz(npz_path) if args.clamp_to_npz_len else None
            runs_pose = map_runs_with_stride_and_clamp(
                runs_raw,
                frame_stride=int(args.frame_stride),
                clamp_len=clamp_len,
            )

            if runs_pose:
                spans[stem] = runs_pose
            else:
                # fallback: full length if we can infer it
                n_raw = len(clip_frames.get(ck, []))
                if n_raw > 0:
                    # map full range with stride
                    full_pose = map_runs_with_stride_and_clamp(
                        [[0, n_raw]],
                        frame_stride=int(args.frame_stride),
                        clamp_len=clamp_len,
                    )
                    if full_pose:
                        spans[stem] = full_pose

    # Write outputs
    _ensure_dir_for_file(args.out_labels)
    _ensure_dir_for_file(args.out_spans)

    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    with open(args.out_spans, "w", encoding="utf-8") as f:
        json.dump(spans, f, indent=2, sort_keys=True)

    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)} fall={fall_total})")
    print(f"[OK] wrote spans  → {args.out_spans}  (videos_with_spans={len(spans)})")
    if missing_clipkey:
        print(f"[warn] {missing_clipkey} stems did not match clip key pattern like adl-01-cam0-rgb / fall-02-cam1-rgb")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
