#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_muvim_labels.py

MUVIM labels + optional fall spans builder.

Inputs
------
- Pose NPZ files under --npz_dir (canonical stems for pipeline keys)
- Optional ZED_RGB.csv with columns: Video, Start, Stop

Outputs
-------
- out_labels JSON: {stem: 0/1}   (0=adl, 1=fall)
- out_spans  JSON: {stem: [[start, end_excl], ...]}  (pose-frame index units)

Key design goals
----------------
1) Use NPZ stems as canonical keys (what downstream code will read).
2) Make span mapping robust:
   - optional padding
   - optional stride mapping
   - optional clamping to NPZ sequence length
3) Always use end-exclusive spans.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Regex helpers to infer / match IDs
_FALL_RE = re.compile(r"(?:^|[^a-z0-9])fall\s*0*([0-9]+)", re.IGNORECASE)
_NONFALL_RE = re.compile(r"(?:^|[^a-z0-9])non\s*fall\s*0*([0-9]+)", re.IGNORECASE)
_VIDEO_RE = re.compile(r"(?:^|[^a-z0-9])video\s*0*([0-9]+)", re.IGNORECASE)


def list_npz_files(npz_dir: str) -> List[str]:
    """Return all NPZ files under a directory (recursive), sorted."""
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))


def infer_label_from_path(p: str) -> int:
    """
    Infer label from file/folder naming conventions.
    Returns:
      1 for fall, 0 for ADL/non-fall

    NonFall must win over Fall (because "NonFall10" contains "Fall").
    """
    s = pathlib.Path(p).stem.lower()
    parent = pathlib.Path(p).parent.name.lower()

    if s.startswith("nonfall") or parent.startswith("nonfall") or _NONFALL_RE.search(s):
        return 0
    if s.startswith("fall") or parent.startswith("fall") or _FALL_RE.search(s):
        return 1
    return 0


def extract_video_id(stem_or_path: str) -> Optional[int]:
    """
    Extract numeric Video id used by ZED_RGB.csv.

    We accept:
      Fall10 / Video10 / ... video__10 ...
    but we avoid wrongly extracting from NonFall.
    """
    s = str(stem_or_path)
    if _NONFALL_RE.search(s):
        return None

    m = _FALL_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    m = _VIDEO_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    return None


def load_spans_from_csv(csv_path: str) -> Dict[int, List[List[int]]]:
    """
    Read ZED_RGB.csv: columns must include Video, Start, Stop.
    Returns:
      {video_id: [[start, stop_excl], ...]}  (still in RAW frame indices)
    """
    spans_by_vid: Dict[int, List[List[int]]] = defaultdict(list)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"[ERR] {csv_path} has no header row")

        required = {"Video", "Start", "Stop"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"[ERR] {csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            try:
                vid = int(row["Video"])
                s = int(float(row["Start"]))
                e = int(float(row["Stop"]))
            except Exception:
                continue

            s = max(0, s)
            if e <= s:
                continue

            # Keep raw for now; we will apply inclusive/stride/pad later
            spans_by_vid[vid].append([s, e])

    # sort each list
    for vid in spans_by_vid:
        spans_by_vid[vid].sort(key=lambda x: (x[0], x[1]))

    return dict(spans_by_vid)


def merge_spans(spans: List[List[int]]) -> List[List[int]]:
    """Merge overlaps/adjacent spans (end-exclusive)."""
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out: List[List[int]] = []
    for s, e in spans:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out


def seq_len_from_npz(path: str) -> Optional[int]:
    """
    Read sequence length (T) from pose NPZ.
    Supports both keys:
      - xy
      - joints
    """
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
            if "joints" in z:
                return int(z["joints"].shape[0])
    except Exception:
        return None
    return None


def map_span_raw_to_pose(
    s_raw: int,
    e_raw: int,
    *,
    stop_inclusive: bool,
    frame_stride: int,
    pad_pre: int,
    pad_post: int,
    clamp_len: Optional[int],
) -> Optional[List[int]]:
    """
    Convert raw span -> pose span.

    Steps:
    1) If stop_inclusive: make it exclusive by +1
    2) Apply padding (pre/post) in raw-frame units
    3) Convert raw->pose indices using stride:
         s_pose = floor(s_raw / k)
         e_pose = ceil(e_raw_excl / k) = (e_raw_excl + k - 1) // k
    4) Clamp to [0, clamp_len] if clamp_len provided
    """
    k = max(1, int(frame_stride))

    s = max(0, int(s_raw) - int(pad_pre))
    e_excl = int(e_raw) + (1 if stop_inclusive else 0)
    e_excl = e_excl + int(pad_post)

    if e_excl <= s:
        return None

    s_pose = s // k
    e_pose = (e_excl + k - 1) // k  # ceil

    if clamp_len is not None:
        s_pose = max(0, min(s_pose, clamp_len))
        e_pose = max(0, min(e_pose, clamp_len))

    if e_pose <= s_pose:
        return None
    return [int(s_pose), int(e_pose)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MUVIM labels and optional fall spans from ZED_RGB.csv.")
    ap.add_argument("--npz_dir", default="data/interim/muvim/pose_npz")
    ap.add_argument("--out_labels", default="configs/labels/muvim.json")
    ap.add_argument("--out_spans", default="configs/labels/muvim_spans.json")

    ap.add_argument("--zed_csv", default=None, help="Path to ZED_RGB.csv. If omitted, spans are not written.")
    ap.add_argument("--pad_pre", type=int, default=0, help="Pad span start by this many raw frames.")
    ap.add_argument("--pad_post", type=int, default=0, help="Pad span end by this many raw frames.")
    ap.add_argument("--frame_stride", type=int, default=1, help="If pose extraction used every k-th raw frame, set k here.")
    ap.add_argument("--stop_inclusive", action="store_true", help="Treat Stop as inclusive (convert to exclusive by +1).")
    ap.add_argument("--clamp_to_npz_len", action="store_true", help="Clamp spans to each NPZ sequence length (best-effort).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    # 1) Default labels from path conventions
    labels: Dict[str, int] = {}
    for p in files:
        stem = pathlib.Path(p).stem
        labels[stem] = infer_label_from_path(p)

    # 2) Optional spans from CSV
    spans_out: Dict[str, List[List[int]]] = {}
    if args.zed_csv:
        spans_by_vid = load_spans_from_csv(args.zed_csv)

        matched = 0
        for p in files:
            stem = pathlib.Path(p).stem

            # Find video id for this pose sequence
            vid = extract_video_id(stem) or extract_video_id(p)
            if vid is None or vid not in spans_by_vid:
                continue

            clamp_len = seq_len_from_npz(p) if args.clamp_to_npz_len else None

            pose_spans: List[List[int]] = []
            for s_raw, e_raw in spans_by_vid[vid]:
                sp = map_span_raw_to_pose(
                    s_raw,
                    e_raw,
                    stop_inclusive=bool(args.stop_inclusive),
                    frame_stride=int(args.frame_stride),
                    pad_pre=int(args.pad_pre),
                    pad_post=int(args.pad_post),
                    clamp_len=clamp_len,
                )
                if sp:
                    pose_spans.append(sp)

            pose_spans = merge_spans(pose_spans)
            if pose_spans:
                spans_out[stem] = pose_spans
                labels[stem] = 1  # if spans exist, this must be a fall clip
                matched += 1

        print(f"[info] loaded spans for {len(spans_by_vid)} Video IDs from {args.zed_csv}")
        print(f"[info] matched spans to {matched} pose sequences (out of {len(files)})")

    # 3) Write outputs
    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    fall_n = sum(1 for v in labels.values() if v == 1)
    adl_n = sum(1 for v in labels.values() if v == 0)
    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)} fall={fall_n} adl={adl_n})")

    if args.zed_csv:
        os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
        with open(args.out_spans, "w", encoding="utf-8") as f:
            json.dump(spans_out, f, indent=2, sort_keys=True)
        print(f"[OK] wrote spans  → {args.out_spans} (videos_with_spans={len(spans_out)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
