#!/usr/bin/env python3
"""
CAUCAFall labels (+ optional fall spans) builder (rewrite).

Writes
------
- --out_labels : JSON {stem: "adl"|"fall"} (video/sequence-level label)
- --out_spans  : JSON {stem: [[start, stop], ...]} (optional; half-open)

Why spans?
----------
CAUCAFall "fall" sequences often contain pre-fall ADL motion. If you label the whole clip
as positive, you contaminate training (walk/stand becomes "fall-ish") and increase false alarms.
If per-frame annotation txts encode fall-vs-adl classes, we can derive fall spans and later
label windows by span overlap.

Inputs
------
- --npz_dir: cleaned pose sequences (.npz) (default: data/interim/caucafall/pose_npz)

Optional per-frame annotation support:
- --ann_glob: pattern with "{stem}" placeholder, e.g. 'data/raw/CAUCAFall/**/{stem}/*.txt'
- --fall_class_id: numeric class id indicating FALL (only if your txts encode action classes)
- --min_run / --gap_fill: run-length cleanup for spans

Important caveat
----------------
Some CAUCAFall per-frame txt files may be *person bounding boxes only* (class=person).
In that case, span derivation will not work (no "fall class"). The script will still
write labels inferred from folder/stem tokens, but spans may be empty.
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
    Safer heuristic than 'fall' substring:
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
    return sorted(glob.glob(ann_glob.format(stem=stem), recursive=True))

def parse_frame_is_fall(txt_path: str, fall_class_id: int) -> Optional[bool]:
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

    # optional gap filling
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
    ap.add_argument("--npz_dir", default="data/interim/caucafall/pose_npz")
    ap.add_argument("--out_labels", default="configs/labels/caucafall.json")
    ap.add_argument("--out_spans", default="configs/labels/caucafall_spans.json")
    ap.add_argument("--ann_glob", default=None,
                    help="Per-frame txt glob with {stem} placeholder, e.g. 'data/raw/CAUCAFall/**/{stem}/*.txt'")
    ap.add_argument("--use_per_frame_action_txt", type=int, default=0,
                    help="Set to 1 to parse per-frame action txt files via --ann_glob. Default 0 uses clip-level labels only.")
    ap.add_argument("--fall_class_id", type=int, default=0)
    ap.add_argument("--min_run", type=int, default=3)
    ap.add_argument("--gap_fill", type=int, default=1)
    ap.add_argument("--print_stats", action="store_true")
    args = ap.parse_args()

    if args.ann_glob and int(args.use_per_frame_action_txt) != 1:
        print("[warn] CAUCAFall: ignoring --ann_glob because --use_per_frame_action_txt=0. "
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
                    flags.append(bool(is_fall))

                # Heuristic guard: if almost every frame is "fall", it's very likely that
                # the txt files are person bounding boxes (YOLO) rather than fall-vs-ADL action labels.
                # In that case, deriving spans will poison training, so we skip spans.
                if len(flags) >= 20:
                    fall_ratio = float(sum(flags)) / float(len(flags))
                    if fall_ratio >= 0.95:
                        suspicious_ann += 1
                        # Keep clip-level label inferred from path; do not force "fall" from spans.
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
                "       If you truly have per-frame action labels, set --fall_class_id correctly or disable this guard by editing make_caucafall_labels.py."
            )

if __name__ == "__main__":
    main()
