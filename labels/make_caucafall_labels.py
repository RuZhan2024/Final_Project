
"""
CAUCAFall – build labels (stem -> "adl"/"fall") and optional per-frame fall spans.

Why this exists
---------------
Some CAUCAFall releases provide per-frame annotation .txt files (often in YOLO
format) where the *class id* indicates "fall" vs "nofall". If you label an
entire "fall video" as positive, then pre-fall / post-fall windows become
positive too, which can inflate positives after windowing and push thresholds
lower / increase false alarms.

This script can optionally derive fall spans from per-frame .txt annotations so
window labels can be computed by span overlap in make_windows.py.

Outputs
-------
1) labels JSON:
   { "<stem>": "fall" | "adl" }

2) spans JSON (optional):
   { "<stem>": [[start, stop], ...] }

Span convention: half-open intervals [start, stop) in *frame index units*
relative to the extraction order (sorted by filename). Use
`--spans_end_exclusive` in make_windows.py when consuming these spans.

Per-frame annotation assumption
-------------------------------
Each frame has a corresponding .txt file that may contain one or more lines.
We interpret a frame as "fall" if ANY line has class_id == fall_class_id.

Common YOLO line format:
  <class_id> <x_center> <y_center> <width> <height>

If your annotation files literally contain the strings "fall"/"nofall" instead
of numeric IDs, this script also supports that.

Usage examples
--------------
# Video-level labels only (legacy behaviour)
python labels/make_caucafall_labels.py --npz_dir data/interim/caucafall/pose_npz --out_labels configs/labels/caucafall.json

# Labels + spans (recommended if you have per-frame .txt annotations)
python labels/make_caucafall_labels.py \
  --npz_dir data/interim/caucafall/pose_npz \
  --ann_glob 'data/raw/caucafall/frames/{stem}/*.txt' \
  --out_labels configs/labels/caucafall.json \
  --out_spans  configs/labels/caucafall_spans.json \
  --fall_class_id 0 --min_run 3 --gap_fill 1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple


def list_npzs(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))


def tokenise(s: str) -> List[str]:
    # split by non-alphanumeric boundaries
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def infer_label_from_path(npz_path: str) -> str:
    """
    Safer heuristic than `'fall' in name`:
    - If tokens include nonfall/nofall/adl -> adl
    - Else if tokens include fall -> fall
    - Else -> adl
    """
    p = pathlib.Path(npz_path)
    stem_tokens = tokenise(p.stem)
    parent_tokens = tokenise(p.parent.name if p.parent else "")
    toks = set(stem_tokens + parent_tokens + [t for part in p.parts for t in tokenise(part)])

    if any(t in toks for t in ("nonfall", "nofall", "adl", "normal")):
        return "adl"
    if "fall" in toks:
        return "fall"
    return "adl"


def expand_ann_glob(ann_glob: str, stem: str) -> List[str]:
    """
    ann_glob should include "{stem}" placeholder, e.g.
      data/raw/caucafall/labels/{stem}/*.txt
      data/raw/caucafall/**/{stem}/*.txt
      data/raw/caucafall/**/{stem}/*.txt
    """
    pattern = ann_glob.format(stem=stem)
    return sorted(glob.glob(pattern, recursive=True))


def parse_frame_label(txt_path: str, fall_class_id: int) -> Optional[bool]:
    """
    Return True if frame labeled fall, False if nofall, None if file unreadable.
    Supports:
      - YOLO-like numeric first token (class id)
      - literal 'fall' / 'nofall' tokens anywhere in the file
    """
    try:
        content = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None

    if not content:
        # empty annotation: treat as no fall (common if no object detected)
        return False

    low = content.lower()
    # literal labels
    if "nofall" in low or "nonfall" in low:
        return False
    # careful: "fall" is substring of "nonfall", already handled above
    if re.search(r"\bfall\b", low):
        return True

    # numeric class id case: check first token on each line
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
        if cid == fall_class_id:
            return True

    return False


def bool_runs_to_spans(
    flags: List[bool],
    min_run: int = 1,
    gap_fill: int = 0,
) -> List[List[int]]:
    """
    Convert per-frame boolean flags to half-open spans [start, stop).

    gap_fill: fill gaps of up to this many False frames inside a True run.
    min_run:  minimum run length (in frames) to keep.
    """
    if not flags:
        return []

    # optional gap filling
    if gap_fill > 0:
        filled = flags[:]
        i = 0
        n = len(filled)
        while i < n:
            if filled[i]:
                i += 1
                continue
            # find a false gap
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
        stop = j  # half-open
        if (stop - start) >= max(1, int(min_run)):
            spans.append([start, stop])
        i = stop
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="data/interim/caucafall/pose_npz")
    ap.add_argument("--out_labels", required=True, help="configs/labels/caucafall.json")
    ap.add_argument("--out_spans", default=None, help="configs/labels/caucafall_spans.json (optional)")
    ap.add_argument(
        "--ann_glob",
        default=None,
        help="Glob pattern with {stem} placeholder pointing to per-frame .txt annotations, e.g. 'data/raw/caucafall/**/{stem}/*.txt'",
    )
    ap.add_argument("--fall_class_id", type=int, default=0, help="Numeric class id that means FALL in per-frame txt.")
    ap.add_argument("--min_run", type=int, default=3, help="Minimum consecutive fall frames to form a span.")
    ap.add_argument("--gap_fill", type=int, default=1, help="Fill short gaps (False runs) up to this many frames.")
    ap.add_argument("--print_stats", action="store_true", help="Print simple stats about derived spans.")
    args = ap.parse_args()

    npzs = list_npzs(args.npz_dir)
    if not npzs:
        raise SystemExit(f"[ERR] No NPZs under {args.npz_dir}")

    labels: Dict[str, str] = {}
    spans_out: Dict[str, List[List[int]]] = {}

    for npz_path in npzs:
        stem = pathlib.Path(npz_path).stem

        # default heuristic label
        lab = infer_label_from_path(npz_path)

        # optional: derive spans from per-frame annotations
        if args.ann_glob and args.out_spans:
            txts = expand_ann_glob(args.ann_glob, stem)
            if txts:
                # interpret sorted txts as the same temporal order as extraction
                flags: List[bool] = []
                for tpath in txts:
                    is_fall = parse_frame_label(tpath, args.fall_class_id)
                    flags.append(bool(is_fall))  # None treated as False
                spans = bool_runs_to_spans(flags, min_run=args.min_run, gap_fill=args.gap_fill)
                if spans:
                    spans_out[stem] = spans
                    lab = "fall"  # spans imply this sequence has fall frames

        labels[stem] = lab

    # write outputs
    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w") as f:
        json.dump(labels, f, indent=2)

    fall_n = sum(1 for v in labels.values() if v == "fall")
    adl_n = sum(1 for v in labels.values() if v == "adl")
    print(f"[OK] wrote {len(labels)} labels → {args.out_labels} (fall={fall_n}, adl={adl_n})")

    if args.out_spans:
        os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
        with open(args.out_spans, "w") as f:
            json.dump(spans_out, f, indent=2)
        print(f"[OK] wrote {len(spans_out)} span entries → {args.out_spans} (half-open [start, stop))")

        if args.print_stats and spans_out:
            # simple stats: average span length and how many spans per video
            lens = [stop - start for sps in spans_out.values() for start, stop in sps]
            nsp = [len(sps) for sps in spans_out.values()]
            print(f"[stats] spans/videos: min={min(nsp)}, median={sorted(nsp)[len(nsp)//2]}, max={max(nsp)}")
            print(f"[stats] span length (frames): min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)}")


if __name__ == "__main__":
    main()
