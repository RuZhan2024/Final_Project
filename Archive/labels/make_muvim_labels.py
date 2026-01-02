#!/usr/bin/env python3
"""
MUVIM labels + fall spans builder (rewrite).

Reads
-----
- Pose NPZs under --npz_dir (default: data/interim/muvim/pose_npz)
- Optional ZED_RGB.csv under --zed_csv, providing frame-level fall intervals.

Writes
------
- --out_labels : JSON mapping {stem: "adl"|"fall"}
- --out_spans  : JSON mapping {stem: [[start, stop], ...]} with half-open spans [start, stop)

Key improvements vs legacy script
---------------------------------
1) More robust video-id extraction from stems/paths.
2) Optional clamping of spans to each pose sequence length (prevents out-of-range spans).
3) Optional frame_stride mapping if pose was extracted with frame skipping.

CSV assumptions
---------------
ZED_RGB.csv must contain columns: Video, Start, Stop (case-sensitive in the CSV header).
We treat [Start, Stop) as half-open by default. If your CSV uses inclusive Stop, pass --stop_inclusive.

Span conventions
----------------
- Spans are half-open [start, stop) in pose-frame index units.
- If you extracted every k-th frame, set --frame_stride=k to map raw frame -> pose frame.
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

_FALL_RE = re.compile(r"(?:^|[^a-z0-9])fall\s*0*([0-9]+)", re.IGNORECASE)
_NONFALL_RE = re.compile(r"(?:^|[^a-z0-9])non\s*fall\s*0*([0-9]+)", re.IGNORECASE)
_VIDEO_RE = re.compile(r"(?:^|[^a-z0-9])video\s*0*([0-9]+)", re.IGNORECASE)

def list_npz_files(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))

def infer_label_from_path(p: str) -> str:
    """Infer fall/adl from conventional folder or stem prefixes."""
    s = pathlib.Path(p).stem.lower()
    parent = pathlib.Path(p).parent.name.lower()

    # Nonfall must win over fall
    if s.startswith("nonfall") or parent.startswith("nonfall") or _NONFALL_RE.search(s):
        return "adl"
    if s.startswith("fall") or parent.startswith("fall") or _FALL_RE.search(s):
        return "fall"
    return "adl"

def extract_video_id(stem_or_path: str) -> Optional[int]:
    """
    Extract numeric video id used by ZED_RGB.csv.
    Accept 'Fall10', 'Video10' patterns; avoid matching 'NonFall10' as fall.
    """
    s = str(stem_or_path)
    m = _NONFALL_RE.search(s)
    if m:
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
    spans_by_vid: Dict[int, List[List[int]]] = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
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
            if s < 0:
                s = 0
            if e <= s:
                continue
            spans_by_vid[vid].append([s, e])

    for vid in spans_by_vid:
        spans_by_vid[vid].sort(key=lambda x: (x[0], x[1]))
    return dict(spans_by_vid)

def merge_spans(spans: List[List[int]]) -> List[List[int]]:
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out: List[List[int]] = []
    for s, e in spans:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def apply_pad_stride_clamp(
    spans: List[List[int]],
    *,
    pad_pre: int,
    pad_post: int,
    frame_stride: int,
    stop_inclusive: bool,
    clamp_len: Optional[int],
) -> List[List[int]]:
    stride = max(1, int(frame_stride))
    out: List[List[int]] = []
    for s, e in spans:
        if stop_inclusive:
            e = e + 1
        s2 = max(0, int(s) - int(pad_pre))
        e2 = int(e) + int(pad_post)
        # map to pose frame index
        s2 = s2 // stride
        e2 = e2 // stride
        if clamp_len is not None:
            s2 = max(0, min(s2, clamp_len))
            e2 = max(0, min(e2, clamp_len))
        if e2 > s2:
            out.append([s2, e2])
    return merge_spans(out)

def seq_len_from_npz(path: str) -> Optional[int]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser(description="Build MUVIM labels and optional fall spans from ZED_RGB.csv.")
    ap.add_argument("--npz_dir", default="data/interim/muvim/pose_npz")
    ap.add_argument("--out_labels", default="configs/labels/muvim.json")
    ap.add_argument("--out_spans", default="configs/labels/muvim_spans.json")
    ap.add_argument("--zed_csv", default=None, help="Path to ZED_RGB.csv. If omitted, spans are not written.")
    ap.add_argument("--pad_pre", type=int, default=0)
    ap.add_argument("--pad_post", type=int, default=0)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--stop_inclusive", action="store_true",
                    help="Treat Stop column as inclusive (convert to exclusive by +1).")
    ap.add_argument("--clamp_to_npz_len", action="store_true")
    args = ap.parse_args()

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    stems: List[str] = []
    labels: Dict[str, str] = {}

    for p in files:
        stem = pathlib.Path(p).stem
        stems.append(stem)
        labels[stem] = infer_label_from_path(p)

    spans_out: Dict[str, List[List[int]]] = {}
    if args.zed_csv:
        spans_by_vid = load_spans_from_csv(args.zed_csv)

        matched = 0
        for p in files:
            stem = pathlib.Path(p).stem
            vid = extract_video_id(stem) or extract_video_id(p)
            if vid is None:
                continue
            if vid not in spans_by_vid:
                continue

            clamp_len = seq_len_from_npz(p) if args.clamp_to_npz_len else None
            sps = apply_pad_stride_clamp(
                spans_by_vid[vid],
                pad_pre=args.pad_pre,
                pad_post=args.pad_post,
                frame_stride=args.frame_stride,
                stop_inclusive=args.stop_inclusive,
                clamp_len=clamp_len,
            )
            if sps:
                spans_out[stem] = sps
                labels[stem] = "fall"
                matched += 1

        print(f"[info] loaded spans for {len(spans_by_vid)} video IDs from {args.zed_csv}")
        print(f"[info] matched spans to {matched} pose sequences (out of {len(files)})")

    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    fall_n = sum(1 for v in labels.values() if v == "fall")
    adl_n = sum(1 for v in labels.values() if v == "adl")
    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)}, fall={fall_n}, adl={adl_n})")

    if args.zed_csv:
        os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
        with open(args.out_spans, "w", encoding="utf-8") as f:
            json.dump(spans_out, f, indent=2)
        print(f"[OK] wrote spans  → {args.out_spans}  (videos_with_spans={len(spans_out)})")

if __name__ == "__main__":
    main()
