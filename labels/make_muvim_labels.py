#!/usr/bin/env python3
"""
MUVIM labels + fall spans builder (rewrite).

Critical updates:
- More robust extract_video_id() supporting: F01_Cam1, F_01, NF01, etc.
- Verbose warnings for read/parse failures.
- Final summary stats always printed.
- CSV reading uses encoding='utf-8', errors='ignore'.
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
from typing import Dict, List, Optional

import numpy as np

# Existing patterns
_FALL_RE = re.compile(r"(?:^|[^a-z0-9])fall\s*0*([0-9]+)", re.IGNORECASE)
_NONFALL_RE = re.compile(r"(?:^|[^a-z0-9])non\s*fall\s*0*([0-9]+)", re.IGNORECASE)
_VIDEO_RE = re.compile(r"(?:^|[^a-z0-9])video\s*0*([0-9]+)", re.IGNORECASE)

# New robust short patterns (no spaces)
# Matches: F01, F_01, F-01, F01_Cam1, etc.
_F_SHORT_RE = re.compile(r"(?:^|[^a-z0-9])f\s*[_-]?\s*0*([0-9]+)\b", re.IGNORECASE)
# Matches: NF01, NF_01, etc.
_NF_SHORT_RE = re.compile(r"(?:^|[^a-z0-9])nf\s*[_-]?\s*0*([0-9]+)\b", re.IGNORECASE)


def list_npz_files(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))


def infer_label_from_path(p: str) -> str:
    """Infer fall/adl from conventional folder or stem prefixes."""
    s = pathlib.Path(p).stem.lower()
    parent = pathlib.Path(p).parent.name.lower()

    # Nonfall must win over fall
    if s.startswith("nonfall") or parent.startswith("nonfall") or _NONFALL_RE.search(s) or _NF_SHORT_RE.search(s):
        return "adl"
    if s.startswith("fall") or parent.startswith("fall") or _FALL_RE.search(s) or _F_SHORT_RE.search(s):
        return "fall"
    return "adl"


def extract_video_id(stem_or_path: str) -> Optional[int]:
    """
    Extract numeric video id used by ZED_RGB.csv.
    Supports:
      - Fall10 / Fall 10
      - Video10 / Video 10
      - F01 / F_01 / F01_Cam1
    Non-fall patterns (NonFall / NF..) return None.
    """
    s = str(stem_or_path)

    # Non-fall must win
    if _NONFALL_RE.search(s) or _NF_SHORT_RE.search(s) or re.search(r"\bno\s*fall\b", s, re.IGNORECASE):
        return None

    # Short fall IDs (F01...)
    m = _F_SHORT_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    # FallNN / VideoNN patterns
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


def load_spans_from_csv(csv_path: str, *, verbose: bool = False) -> Dict[int, List[List[int]]]:
    spans_by_vid: Dict[int, List[List[int]]] = defaultdict(list)
    try:
        with open(csv_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SystemExit(f"[ERR] {csv_path} has no header row")

            required = {"Video", "Start", "Stop"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise SystemExit(f"[ERR] {csv_path} missing columns: {sorted(missing)}")

            bad_rows = 0
            for row in reader:
                try:
                    vid = int(row["Video"])
                    s = int(float(row["Start"]))
                    e = int(float(row["Stop"]))
                except Exception:
                    bad_rows += 1
                    continue
                if s < 0:
                    s = 0
                if e <= s:
                    continue
                spans_by_vid[vid].append([s, e])

            if verbose and bad_rows:
                print(f"[warn] skipped {bad_rows} malformed CSV rows in {csv_path}")
    except FileNotFoundError:
        raise SystemExit(f"[ERR] CSV not found: {csv_path}")
    except Exception as e:
        raise SystemExit(f"[ERR] failed reading CSV {csv_path}: {e}")

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


def seq_len_from_npz(path: str, *, verbose: bool = False) -> Optional[int]:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
    except Exception as e:
        if verbose:
            print(f"[warn] could not read npz length: {path} ({e})")
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
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    labels: Dict[str, str] = {}
    stems_scanned = 0

    for p in files:
        stem = pathlib.Path(p).stem
        stems_scanned += 1
        labels[stem] = infer_label_from_path(p)

    spans_out: Dict[str, List[List[int]]] = {}
    matched_spans = 0
    vid_miss = 0

    if args.zed_csv:
        spans_by_vid = load_spans_from_csv(args.zed_csv, verbose=args.verbose)

        for p in files:
            stem = pathlib.Path(p).stem
            vid = extract_video_id(stem) or extract_video_id(p)
            if vid is None:
                vid_miss += 1
                continue
            if vid not in spans_by_vid:
                continue

            clamp_len = seq_len_from_npz(p, verbose=args.verbose) if args.clamp_to_npz_len else None
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
                matched_spans += 1

        print(f"[info] loaded spans for {len(spans_by_vid)} video IDs from {args.zed_csv}")
        print(f"[info] matched spans to {matched_spans} pose sequences (out of {len(files)})")
        if args.verbose and vid_miss:
            print(f"[warn] could not extract a video id for {vid_miss} sequences (stem/path pattern mismatch)")

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

    # Required verification stats
    total_npz = len(files)
    matched_labels = len(labels)
    missing = total_npz - matched_labels
    print(f"[summary] npz_total={total_npz}  matched_labels={matched_labels}  missing_or_skipped={missing}")

if __name__ == "__main__":
    main()
