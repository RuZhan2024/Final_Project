#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_caucafall_labels_from_frames.py

CAUCAFall labels + spans that match pose NPZ stems.

Problem this script solves
--------------------------
Your pose pipeline names sequences like:
  Subject.1__Fall_backwards__ec0a78a8.npz

But raw data is organized like:
  raw_root/Subject.1/Fall backwards/*.txt  (per-frame YOLO labels)
  raw_root/Subject.1/Walk/*.txt

For window generation, label keys MUST match pose NPZ stems.
So we iterate pose NPZ files (canonical keys), then locate the raw action folder.

Span generation
---------------
We read per-frame YOLO txt files in the action folder.
A frame is considered "fall" if any line begins with the fall class id.

We output:
  - out_labels: {pose_stem: 0/1}
  - out_spans : {pose_stem: [[start, end_excl], ...]}  (pose index space)

We also support:
  - --frame_stride: map raw-frame spans -> pose indices
  - --clamp_to_npz_len: clamp spans to pose length (best-effort)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    """
    Normalize a string for robust folder matching:
      'Fall backwards' -> 'fall_backwards'
      'Fall_backwards' -> 'fall_backwards'
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _last_int_group(name: str) -> Optional[int]:
    """Extract last integer group from basename (used for ordering frames)."""
    stem = os.path.splitext(os.path.basename(name))[0]
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


def _is_fall_frame(txt_path: str, fall_class_id: int) -> bool:
    """
    A frame is "fall" if any YOLO line has class_id == fall_class_id.
    We only parse the first token (class id).
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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


def spans_from_action_dir(
    action_dir: str,
    fall_class_id_default: int,
    min_run: int,
    gap_fill: int,
) -> List[List[int]]:
    """
    Build raw-frame fall spans from an action directory.

    - Determine fall class id from classes.txt if present.
    - Read all per-frame txt files (excluding classes.txt).
    - Sort by numeric index.
    - Build boolean flags.
    - Gap-fill and convert to runs.
    """
    # Determine fall class id
    classes_path = os.path.join(action_dir, "classes.txt")
    mapping = _read_classes_txt(classes_path)

    fall_id = fall_class_id_default
    for k, v in mapping.items():
        if v == "fall":
            fall_id = k
            break

    txts = [p for p in glob.glob(os.path.join(action_dir, "*.txt")) if os.path.basename(p).lower() != "classes.txt"]
    txts = [p for p in txts if _last_int_group(p) is not None]
    if not txts:
        return []

    txts.sort(key=_last_int_group)

    flags: List[bool] = [_is_fall_frame(p, fall_id) for p in txts]
    flags = _gap_fill(flags, gap_fill)
    runs = _runs(flags, min_run)
    return [[a, b] for (a, b) in runs]


def _find_raw_root(raw_root: str, verbose: bool) -> str:
    """
    Allow passing either:
      - the CAUCAFall root (contains Subject.*)
      - or its parent (contains a folder that contains Subject.*)
    """
    subj = [d for d in glob.glob(os.path.join(raw_root, "Subject.*")) if os.path.isdir(d)]
    if subj:
        return raw_root

    for child in [d for d in glob.glob(os.path.join(raw_root, "*")) if os.path.isdir(d)]:
        subj = [d for d in glob.glob(os.path.join(child, "Subject.*")) if os.path.isdir(d)]
        if subj:
            if verbose:
                print(f"[warn] Subject.* not found directly under {raw_root}; using nested root: {child}")
            return child

    return raw_root


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Path to data/raw/CAUCAFall")
    ap.add_argument("--npz_dir", required=True, help="Path to interim/caucafall/pose_npz (canonical stems)")
    ap.add_argument("--out_labels", required=True)
    ap.add_argument("--out_spans", required=True)

    ap.add_argument("--min_run", type=int, default=2)
    ap.add_argument("--gap_fill", type=int, default=2)
    ap.add_argument("--fall_class_id", type=int, default=1)
    ap.add_argument("--frame_stride", type=int, default=1, help="Pose sampling stride relative to raw frames.")
    ap.add_argument("--clamp_to_npz_len", action="store_true", help="Clamp pose spans to NPZ length.")
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

        subj = parts[0]         # Subject.1
        action_token = parts[1] # Fall_backwards

        amap = subj_actions.get(subj)
        if not amap:
            missing_map += 1
            continue

        action_dir = amap.get(_norm(action_token))
        if not action_dir:
            missing_map += 1
            continue

        # Label is from folder name: "Fall ..." folders are positives
        is_fall = os.path.basename(action_dir).lower().startswith("fall")
        labels[stem] = 1 if is_fall else 0

        if is_fall:
            raw_spans = spans_from_action_dir(
                action_dir=action_dir,
                fall_class_id_default=int(args.fall_class_id),
                min_run=int(args.min_run),
                gap_fill=int(args.gap_fill),
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
                # fallback: if no fall frames detected, use full length based on txt count
                txts = [p for p in glob.glob(os.path.join(action_dir, "*.txt")) if os.path.basename(p).lower() != "classes.txt"]
                txts = [p for p in txts if _last_int_group(p) is not None]
                if txts:
                    txts.sort(key=_last_int_group)
                    full_pose = map_spans_raw_to_pose(
                        [[0, len(txts)]],
                        frame_stride=int(args.frame_stride),
                        clamp_len=clamp_len,
                    )
                    if full_pose:
                        spans[stem] = full_pose

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
