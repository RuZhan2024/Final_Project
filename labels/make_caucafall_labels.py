#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_caucafall_labels.py

CAUCAFall labels + optional spans that match pose NPZ stems.

What it does
------------
1) Iterate pose NPZ files in --npz_dir. Their stems are the canonical keys.
2) Map each stem -> raw action directory under --raw_root/Subject.*/<Action Folder>
3) Video-level label:
      1 if action folder name starts with "Fall"
      0 otherwise
4) If --use_per_frame_action_txt 1 and label==1:
   Read per-frame .txt files to derive fall start/stop spans.
   Output spans in pose-index space.

Frame txt formats supported
---------------------------
A) Action-label frames:
   - Each frame txt contains one or more lines, where the FIRST token is a class id.
   - classes.txt (optional) maps class_id -> class_name (one name per line).
   - If any line begins with fall_class_id => that frame is fall.

B) YOLO bbox-only frames:
   - Lines are: class xc yc w h (normalized floats)
   - No "fall" class exists.
   - We can (optionally) estimate fall frames with bbox geometry (auto mode).

Safety defaults
---------------
- If spans cannot be derived, we DO NOT create a full-clip span by default.
  Use --full_clip_fallback_if_no_fall_frames to restore that behavior.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def _norm(s: str) -> str:
    """Normalize a string for robust folder matching."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _last_int_group(path: str) -> Optional[int]:
    """Extract last integer group from basename (used for ordering frames)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    groups = re.findall(r"(\d+)", stem)
    if not groups:
        return None
    return int(groups[-1])


def _read_classes_txt(path: str) -> Dict[int, str]:
    """
    Read classes.txt if present: each line is a class name, line index is class id.
    Returns mapping {id: name_lower}.
    """
    out: Dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                out[i] = s.lower()
    except FileNotFoundError:
        return out
    return out


def _gap_fill(flags: List[bool], gap_fill: int) -> List[bool]:
    """Fill short False gaps inside True segments."""
    if gap_fill <= 0 or not flags:
        return flags
    flags = flags[:]
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        gap = j - i
        if i > 0 and j < n and flags[i - 1] and flags[j] and gap <= gap_fill:
            for t in range(i, j):
                flags[t] = True
        i = j
    return flags


def _runs(flags: List[bool], min_run: int) -> List[Tuple[int, int]]:
    """Convert boolean list into runs (start, end_excl)."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, b in enumerate(flags):
        if b and start is None:
            start = i
        elif (not b) and start is not None:
            end = i
            if (end - start) >= min_run:
                runs.append((start, end))
            start = None
    if start is not None:
        end = len(flags)
        if (end - start) >= min_run:
            runs.append((start, end))
    return runs


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


def map_spans_raw_to_pose(
    spans: List[List[int]],
    *,
    frame_stride: int,
    clamp_len: Optional[int],
) -> List[List[int]]:
    """
    Map raw spans -> pose spans.

    For [s, e_excl) raw:
      s_pose = floor(s / k)
      e_pose = ceil(e_excl / k) = (e_excl + k - 1)//k
    """
    k = max(1, int(frame_stride))
    out: List[List[int]] = []
    for s, e in spans:
        s_pose = int(s) // k
        e_pose = (int(e) + k - 1) // k

        if clamp_len is not None:
            s_pose = max(0, min(s_pose, clamp_len))
            e_pose = max(0, min(e_pose, clamp_len))

        if e_pose > s_pose:
            out.append([s_pose, e_pose])

    # merge overlaps
    out.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def _peek_first_valid_line(txt_path: str) -> Optional[List[str]]:
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                return s.split()
    except FileNotFoundError:
        return None
    return None


def _detect_frame_mode(sample_tokens: List[str]) -> str:
    """
    Detect whether a frame txt line looks like:
      - "action label": just class id (len=1) or class+anything non-bbox
      - "bbox": len>=5 and tokens 1..4 look like floats in [0,1]
    Returns: "class_id" or "bbox"
    """
    if len(sample_tokens) >= 5:
        try:
            vals = [float(sample_tokens[i]) for i in range(1, 5)]
            if all(0.0 <= v <= 1.0 for v in vals):
                return "bbox"
        except Exception:
            pass
    return "class_id"


def _is_fall_frame_class_id(txt_path: str, fall_class_id: int) -> bool:
    """Frame is fall if ANY line begins with class_id == fall_class_id."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok0 = line.split()[0]
                try:
                    cid = int(tok0)
                except Exception:
                    continue
                if cid == fall_class_id:
                    return True
    except FileNotFoundError:
        return False
    return False


def _read_bbox_wh_largest(txt_path: str) -> Optional[Tuple[float, float]]:
    """
    For bbox txt: pick the largest-area bbox and return (w, h).
    Line format: class xc yc w h
    """
    best: Optional[Tuple[float, float]] = None
    best_area = -1.0
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                try:
                    w = float(p[3]); h = float(p[4])
                except Exception:
                    continue
                area = w * h
                if area > best_area:
                    best_area = area
                    best = (w, h)
    except FileNotFoundError:
        return None
    return best


def _median(xs: List[float]) -> float:
    xs2 = sorted(xs)
    n = len(xs2)
    if n == 0:
        return 0.0
    return xs2[n // 2]


def spans_from_action_dir(
    action_dir: str,
    *,
    fall_class_id_default: int,
    min_run: int,
    gap_fill: int,
    frame_label_mode: str,  # "auto" | "class_id" | "bbox"
    bbox_ar_thr: float,
    bbox_h_drop: float,
) -> List[List[int]]:
    """
    Build raw-frame fall spans from an action directory.
    Returns spans in raw-frame index space: [[start, end_excl], ...]
    """
    classes_path = os.path.join(action_dir, "classes.txt")
    mapping = _read_classes_txt(classes_path)

    # Collect per-frame txt files
    txts = [
        p for p in glob.glob(os.path.join(action_dir, "*.txt"))
        if os.path.basename(p).lower() != "classes.txt"
    ]
    # Prefer numeric ordering if possible
    numeric_txts = [p for p in txts if _last_int_group(p) is not None]
    txts = sorted(numeric_txts, key=_last_int_group) if numeric_txts else sorted(txts)

    if not txts:
        return []

    # Decide mode
    mode = frame_label_mode
    if mode == "auto":
        sample = _peek_first_valid_line(txts[0])
        if sample is None:
            return []
        # If classes.txt explicitly contains "fall", treat as class_id
        if any(v == "fall" for v in mapping.values()):
            mode = "class_id"
        else:
            mode = _detect_frame_mode(sample)

    if mode == "class_id":
        fall_id = fall_class_id_default
        for k, v in mapping.items():
            if v == "fall":
                fall_id = k
                break
        flags = [_is_fall_frame_class_id(p, fall_id) for p in txts]

    elif mode == "bbox":
        # Estimate fall frames from bbox geometry
        whs = [_read_bbox_wh_largest(p) for p in txts]
        hs = [h for wh in whs if wh is not None for h in [wh[1]]]
        if not hs:
            return []
        baseline_h = _median(hs)

        flags: List[bool] = []
        for wh in whs:
            if wh is None:
                flags.append(False)
                continue
            w, h = wh
            ar = (w / h) if h > 1e-9 else 999.0
            is_fall = (ar >= bbox_ar_thr) and (h <= bbox_h_drop * baseline_h)
            flags.append(bool(is_fall))
    else:
        raise ValueError(f"Invalid frame_label_mode: {frame_label_mode}")

    flags = _gap_fill(flags, gap_fill)
    runs = _runs(flags, min_run)
    return [[a, b] for (a, b) in runs]


def _find_raw_root(raw_root: str, verbose: bool) -> str:
    """Allow passing either CAUCAFall root (contains Subject.*) or its parent."""
    subj = [d for d in glob.glob(os.path.join(raw_root, "Subject.*")) if os.path.isdir(d)]
    if subj:
        return raw_root

    for child in [d for d in glob.glob(os.path.join(raw_root, "*")) if os.path.isdir(d)]:
        subj2 = [d for d in glob.glob(os.path.join(child, "Subject.*")) if os.path.isdir(d)]
        if subj2:
            if verbose:
                print(f"[warn] Subject.* not found directly under {raw_root}; using nested root: {child}")
            return child

    return raw_root


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Make CAUCAFall labels (+ optional spans) keyed by pose NPZ stems.")
    ap.add_argument("--raw_root", required=True, help="Path to data/raw/CAUCAFall (contains Subject.*)")
    ap.add_argument("--npz_dir", required=True, help="Path to interim/caucafall/pose_npz (canonical stems)")
    ap.add_argument("--out_labels", required=True)
    ap.add_argument("--out_spans", required=True)

    ap.add_argument("--use_per_frame_action_txt", type=int, default=0, help="1 to derive spans from per-frame txt.")
    ap.add_argument("--min_run", type=int, default=2)
    ap.add_argument("--gap_fill", type=int, default=2)
    ap.add_argument("--fall_class_id", type=int, default=1)
    ap.add_argument("--frame_stride", type=int, default=1, help="Pose sampling stride relative to raw frames.")
    ap.add_argument("--clamp_to_npz_len", action="store_true", help="Clamp pose spans to NPZ length.")
    ap.add_argument("--full_clip_fallback_if_no_fall_frames", action="store_true",
                    help="If enabled: when label==1 but no fall frames detected, use full clip span.")

    ap.add_argument("--frame_label_mode", choices=["auto", "class_id", "bbox"], default="auto",
                    help="How to interpret per-frame txt.")
    ap.add_argument("--bbox_ar_thr", type=float, default=0.95, help="BBox mode: w/h threshold.")
    ap.add_argument("--bbox_h_drop", type=float, default=0.65, help="BBox mode: h <= drop*median_h threshold.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    raw_root = _find_raw_root(args.raw_root, args.verbose)

    # Build lookup: subject -> normalized action name -> action_dir
    subj_dirs = sorted([d for d in glob.glob(os.path.join(raw_root, "Subject.*")) if os.path.isdir(d)])
    subj_actions: Dict[str, Dict[str, str]] = {}

    for subj in subj_dirs:
        subj_name = os.path.basename(subj)
        amap: Dict[str, str] = {}
        for action_dir in sorted([d for d in glob.glob(os.path.join(subj, "*")) if os.path.isdir(d)]):
            action_name = os.path.basename(action_dir)
            amap[_norm(action_name)] = action_dir
        subj_actions[subj_name] = amap

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))
    if not npz_files:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    if args.verbose:
        print(f"[info] npz files: {len(npz_files)}")
        print(f"[info] subjects: {len(subj_dirs)}")

    labels: Dict[str, int] = {}
    spans: Dict[str, List[List[int]]] = {}

    missing_map = 0
    for npz_path in npz_files:
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        parts = stem.split("__")
        if len(parts) < 2:
            if args.verbose:
                print(f"[warn] unexpected stem format, skipping: {stem}")
            continue

        subj = parts[0]           # Subject.1
        action_token = parts[1]   # Fall_left / Walk / etc.

        amap = subj_actions.get(subj)
        if not amap:
            missing_map += 1
            if args.verbose:
                print(f"[warn] no subject folder for {subj} (stem={stem})")
            continue

        action_dir = amap.get(_norm(action_token))
        if not action_dir:
            missing_map += 1
            if args.verbose:
                print(f"[warn] no action folder match for {subj}/{action_token} (stem={stem})")
            continue

        # Video-level label from folder name
        is_fall = os.path.basename(action_dir).lower().startswith("fall")
        labels[stem] = 1 if is_fall else 0

        # Optional spans
        if is_fall and int(args.use_per_frame_action_txt) == 1:
            raw_spans = spans_from_action_dir(
                action_dir=action_dir,
                fall_class_id_default=int(args.fall_class_id),
                min_run=int(args.min_run),
                gap_fill=int(args.gap_fill),
                frame_label_mode=str(args.frame_label_mode),
                bbox_ar_thr=float(args.bbox_ar_thr),
                bbox_h_drop=float(args.bbox_h_drop),
            )

            clamp_len = seq_len_from_npz(npz_path) if args.clamp_to_npz_len else None
            pose_spans = map_spans_raw_to_pose(
                raw_spans,
                frame_stride=int(args.frame_stride),
                clamp_len=clamp_len,
            )

            if pose_spans:
                spans[stem] = pose_spans
            else:
                if args.full_clip_fallback_if_no_fall_frames:
                    # full-clip span in raw-frame space based on number of frame txt files
                    txts = [
                        p for p in glob.glob(os.path.join(action_dir, "*.txt"))
                        if os.path.basename(p).lower() != "classes.txt"
                    ]
                    numeric_txts = [p for p in txts if _last_int_group(p) is not None]
                    txts = sorted(numeric_txts, key=_last_int_group) if numeric_txts else sorted(txts)

                    if txts:
                        full_pose = map_spans_raw_to_pose(
                            [[0, len(txts)]],
                            frame_stride=int(args.frame_stride),
                            clamp_len=clamp_len,
                        )
                        if full_pose:
                            spans[stem] = full_pose
                else:
                    if args.verbose:
                        print(f"[warn] no fall spans detected; spans omitted (stem={stem}, dir={action_dir})")

    os.makedirs(os.path.dirname(args.out_labels), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_spans), exist_ok=True)

    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    with open(args.out_spans, "w", encoding="utf-8") as f:
        json.dump(spans, f, indent=2, sort_keys=True)

    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)})")
    print(f"[OK] wrote spans  → {args.out_spans}  (videos_with_spans={len(spans)})")
    if args.verbose and missing_map:
        print(f"[warn] missing subject/action mapping for {missing_map} npz stems")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
