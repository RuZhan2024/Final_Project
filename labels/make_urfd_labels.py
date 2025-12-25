#!/usr/bin/env python3
"""
URFD labels (+ optional fall spans) builder (rewrite).

Goal
----
Produce:
  - labels JSON: stem -> "fall"/"adl"
  - spans JSON (optional): stem -> [[start, stop], ...] (half-open)

Why spans (optional)?
--------------------
If you can derive fall spans from per-frame annotations, window labeling by overlap
is much more accurate than "whole clip is fall". If your per-frame .txt files
do NOT encode fall-vs-adl classes (e.g., only 'person' bounding boxes), then this
script will still produce labels but spans will likely be empty.

Inputs
------
- --npz_dir : cleaned pose sequences (default: data/interim/urfd/pose_npz)

Optional per-frame annotation support:
- --ann_glob : pattern with "{stem}" placeholder pointing to per-frame .txt files for that stem,
               e.g. 'data/raw/UR_Fall_clips/{stem}/*.txt'
- --fall_class_id : YOLO class id indicating FALL (only if your txts encode action classes)

Span extraction rule
--------------------
A frame is considered "fall" if ANY line in its txt has class_id == fall_class_id,
or the file contains a token 'fall' (but not 'nonfall'/'nofall').
We then convert consecutive fall frames into spans [start, stop) with optional gap filling.

Outputs
-------
- --out_labels (default: configs/labels/urfd.json)
- --out_spans  (default: configs/labels/urfd_spans.json) if --ann_glob is provided
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Dict, List, Optional
import numpy as np


def read_npz_src(npz_path: str) -> str:
    """Read embedded 'src' field from a cleaned pose npz if present."""
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            if "src" in z.files:
                v = z["src"]
                # np arrays can store bytes/object; normalize to str
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8", errors="ignore")
                if hasattr(v, "item"):
                    try:
                        return str(v.item())
                    except Exception:
                        return str(v)
                return str(v)
    except Exception:
        return ""
    return ""

def list_npz_files(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))

def tokenise(s: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]

def infer_label_from_path(npz_path: str) -> str:
    """
    Robust label inference:
      - if tokens contain nonfall/nofall/adl -> adl
      - elif tokens contain fall -> fall
      - else -> adl
    """
    p = pathlib.Path(npz_path)
    toks = set()
    for part in p.parts:
        toks.update(tokenise(part))
    toks.update(tokenise(p.stem))

    if any(t in toks for t in ("nonfall", "nofall", "adl", "normal")):
        return "adl"
    if "fall" in toks:
        return "fall"
    return "adl"

def expand_ann_glob(ann_glob: str, stem: str) -> List[str]:
    pattern = ann_glob.format(stem=stem)
    return sorted(glob.glob(pattern, recursive=True))

def parse_frame_is_fall(txt_path: str, fall_class_id: int) -> Optional[bool]:
    """
    Return True if frame labeled fall, False otherwise, None if unreadable.
    Supports:
      - literal tokens 'fall' / 'nofall' / 'nonfall'
      - YOLO numeric class id as first token per line
    """
    try:
        content = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None

    if not content:
        return False

    low = content.lower()
    if "nofall" in low or "nonfall" in low:
        return False
    if re.search(r"\bfall\b", low):
        return True

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid == int(fall_class_id):
            return True
    return False

def bool_runs_to_spans(flags: List[bool], min_run: int, gap_fill: int) -> List[List[int]]:
    if not flags:
        return []

    # fill short gaps inside fall segments
    if gap_fill > 0:
        filled = flags[:]
        i = 0
        n = len(filled)
        while i < n:
            if filled[i]:
                i += 1
                continue
            j = i
            while j < n and not filled[j]:
                j += 1
            gap_len = j - i
            left_true = (i - 1 >= 0 and filled[i - 1])
            right_true = (j < n and filled[j])
            if left_true and right_true and gap_len <= gap_fill:
                for k in range(i, j):
                    filled[k] = True
            i = j
        flags = filled

    spans: List[List[int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        start = i
        j = i
        while j < n and flags[j]:
            j += 1
        stop = j
        if (stop - start) >= max(1, int(min_run)):
            spans.append([start, stop])
        i = stop
    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="data/interim/urfd/pose_npz")
    ap.add_argument("--out_labels", default="configs/labels/urfd.json")
    ap.add_argument("--out_spans", default="configs/labels/urfd_spans.json")
    ap.add_argument("--ann_glob", default=None,
                    help="Per-frame annotation glob with {stem} placeholder. NOTE: DO NOT point this at YOLO bbox .txt files.")
    ap.add_argument("--use_per_frame_action_txt", type=int, default=0,
                    help="Set to 1 only if your per-frame .txt truly encodes FALL/ADL action class IDs (NOT bboxes).")
    ap.add_argument("--fall_class_id", type=int, default=0)
    ap.add_argument("--min_run", type=int, default=3)
    ap.add_argument("--gap_fill", type=int, default=1)
    ap.add_argument("--print_stats", action="store_true")
    args = ap.parse_args()

    if args.ann_glob and int(args.use_per_frame_action_txt) != 1:
        print("[warn] URFD: ignoring --ann_glob because --use_per_frame_action_txt=0. "
              "This is the SAFE default (YOLO bbox .txt are not fall spans).")

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    labels: Dict[str, str] = {}
    spans_out: Dict[str, List[List[int]]] = {}
    suspicious_ann = 0

    for p in files:
        stem = pathlib.Path(p).stem
        src = read_npz_src(str(p))
        lab = infer_label_from_path(src or str(p))

        if int(args.use_per_frame_action_txt) == 1 and args.ann_glob:
            txts = expand_ann_glob(args.ann_glob, stem)
            if txts:
                flags: List[bool] = []
                for t in txts:
                    is_fall = parse_frame_is_fall(t, args.fall_class_id)
                    flags.append(bool(is_fall))  # None -> False

                # Guard: URFD per-frame *.txt are often person bounding boxes (YOLO), not action labels.
                # If almost every frame is flagged as fall, span-derivation is almost certainly wrong.
                if len(flags) >= 20:
                    fall_ratio = float(sum(flags)) / float(len(flags))
                    if fall_ratio >= 0.95:
                        suspicious_ann += 1
                        txts = []

                if txts:
                    spans = bool_runs_to_spans(flags, min_run=args.min_run, gap_fill=args.gap_fill)
                    if spans:
                        spans_out[stem] = spans
                        lab = "fall"

        labels[stem] = lab

    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    fall_n = sum(1 for v in labels.values() if v == "fall")
    adl_n = sum(1 for v in labels.values() if v == "adl")
    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)}, fall={fall_n}, adl={adl_n})")

    if int(args.use_per_frame_action_txt) == 1 and args.ann_glob:
        os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
        with open(args.out_spans, "w", encoding="utf-8") as f:
            json.dump(spans_out, f, indent=2)
        print(f"[OK] wrote spans  → {args.out_spans}  (videos_with_spans={len(spans_out)})")

        if args.print_stats and spans_out:
            lens = [e - s for sps in spans_out.values() for s, e in sps]
            nsp = [len(sps) for sps in spans_out.values()]
            print(f"[stats] spans/videos: min={min(nsp)}, median={sorted(nsp)[len(nsp)//2]}, max={max(nsp)}")
            print(f"[stats] span length (frames): min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)}")

        if not spans_out:
            print("[warn] no spans were derived; this can happen if per-frame txt files do not encode fall-vs-adl classes.")

        if suspicious_ann:
            print(
                f"[warn] skipped span-derivation for {suspicious_ann} videos because per-frame txts looked like bounding boxes (almost all frames flagged as fall).\n"
                "       If you truly have per-frame action labels, set --fall_class_id correctly or disable this guard by editing make_urfd_labels.py."
            )

if __name__ == "__main__":
    main()
