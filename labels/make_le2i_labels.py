#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_le2i_labels.py

LE2i labels + fall spans builder (clean + teach version).

What this script does
---------------------
1) Reads pose sequence NPZ stems from --npz_dir (canonical keys for the pipeline).
2) For each stem, tries to find the matching LE2i annotation txt file.
3) Parses fall start/end indices from annotation file.
4) Converts raw-video indices -> pose indices using --frame_stride.
5) Writes:
     - labels JSON: {stem: 0/1}
     - spans  JSON: {stem: [[start, end_excl], ...]}  (only for positive clips)

Why labels are keyed by NPZ stem
--------------------------------
Your window builder and split maker rely on stem keys matching pose NPZ filenames.
So we treat NPZ stems as "truth keys" and map everything else to those.

Span convention
---------------
- Always end-exclusive: [start, end_excl)
- In pose-frame index units.
- If your extraction sampled every k-th raw frame (frame_stride=k):
    start_pose = floor(start_raw / k)
    end_pose   = ceil(end_raw_excl / k) = (end_raw_excl + k - 1) // k

This avoids spans shrinking too much when stride > 1.

Common LE2i annotation messiness
--------------------------------
LE2i annotation files often contain many integers per line (e.g., bbox dumps).
We only accept a line with:
- exactly 1 integer (we collect two such lines: start then end), OR
- exactly 2 integers (start end)
and ignore lines with >2 integers to avoid reading bbox numbers as spans.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple


# -----------------------------
# 1) Helpers: discover stems
# -----------------------------
def list_npz_stems(npz_dir: str) -> List[str]:
    """
    Return pose NPZ stems in deterministic order.

    We scan recursively because your pose/preprocess pipelines often put
    sequences under nested folders.
    """
    files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))
    return [pathlib.Path(p).stem for p in files]


def read_seq_len(npz_dir: str, stem: str) -> Optional[int]:
    """
    Best-effort sequence length read from the NPZ:
    - prefers 'xy'
    - falls back to 'joints' if present

    Why:
    - some stages write xy, others write joints (alias),
      so supporting both prevents mismatches.
    """
    matches = glob.glob(os.path.join(npz_dir, "**", stem + ".npz"), recursive=True)
    if not matches:
        return None
    p = matches[0]
    try:
        import numpy as np
        with np.load(p, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
            if "joints" in z:
                return int(z["joints"].shape[0])
    except Exception:
        return None
    return None


# -----------------------------
# 2) Extract scene + video id
# -----------------------------
def stem_to_scene_and_vid(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract (scene, video_id) from a stem.

    Your stems typically start with the scene name:
      Coffee_room_01__Videos__video__10_
    so:
      scene = Coffee_room_01

    Video id patterns in the stem are inconsistent; we search robustly:
      - "__video__10" or "__Video__10"
      - "video (10)" or "video_10" etc.
    """
    toks = stem.split("__")
    scene = toks[0] if toks else None

    s = stem.lower()

    # Strong patterns first
    m = re.search(r"__video__\(?(\d+)\)?", s)
    if m:
        return scene, int(m.group(1))

    m = re.search(r"\bvideo\b[^0-9]*\(?(\d+)\)?", s)
    if m:
        return scene, int(m.group(1))

    return scene, None


# -----------------------------
# 3) Find annotation file
# -----------------------------
_CANDIDATE_PATTERNS = [
    "video ({i}).txt", "Video ({i}).txt",
    "video({i}).txt",  "Video({i}).txt",
    "video_{i}.txt",   "Video_{i}.txt",
    "video {i}.txt",   "Video {i}.txt",
]


def find_annotation_file(raw_root: str, scene: str, vid_idx: int) -> Optional[str]:
    """
    LE2i stores annotations under:
      raw_root/<scene>/Annotation_files/*.txt

    But filenames vary across releases, so we try:
    1) a set of deterministic name patterns
    2) a fuzzy regex fallback
    """
    ann_dir = os.path.join(raw_root, scene, "Annotation_files")
    if not os.path.isdir(ann_dir):
        return None

    # (1) deterministic attempts
    for pat in _CANDIDATE_PATTERNS:
        p = os.path.join(ann_dir, pat.format(i=vid_idx))
        if os.path.isfile(p):
            return p

    # (2) fuzzy fallback: any txt whose basename contains this video id
    for p in glob.glob(os.path.join(ann_dir, "*.txt")):
        name = os.path.basename(p).lower()
        if re.search(rf"video\s*[(_ ]?{vid_idx}[)_ ]?\.txt$", name):
            return p

    return None


# -----------------------------
# 4) Parse span from annotation file
# -----------------------------
def parse_span_txt(txt_path: str) -> List[List[int]]:
    """
    Return fall spans from a LE2i annotation file (usually one span).

    We ignore lines with >2 integers to avoid bbox dumps.
    We prefer early lines for speed and robustness.

    Returns:
      [[start_raw, end_raw]]  (raw indices, end may be inclusive/exclusive depending on flag)
      or [] if parsing fails
    """
    try:
        lines = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    singles: List[int] = []
    s: Optional[int] = None
    e: Optional[int] = None

    # Scan early lines first
    for line in lines[:80]:
        ints = [int(x) for x in re.findall(r"-?\d+", line)]
        if not ints:
            continue
        if len(ints) == 1:
            singles.append(ints[0])
            if len(singles) >= 2:
                s, e = singles[0], singles[1]
                break
        elif len(ints) == 2:
            s, e = ints[0], ints[1]
            break
        else:
            continue  # likely bbox dump

    if s is None or e is None:
        # last resort: take the first two ints seen in first 10 lines
        nums: List[int] = []
        for line in lines[:10]:
            nums.extend(int(x) for x in re.findall(r"-?\d+", line))
            if len(nums) >= 2:
                break
        if len(nums) < 2:
            return []
        s, e = nums[0], nums[1]

    # basic sanity
    if s < 0:
        s = 0
    if e <= s:
        return []

    return [[int(s), int(e)]]


# -----------------------------
# 5) Convert raw span -> pose span
# -----------------------------
def raw_span_to_pose_span(
    s_raw: int,
    e_raw: int,
    *,
    end_inclusive: bool,
    frame_stride: int,
    clamp_len: Optional[int],
) -> Optional[List[int]]:
    """
    Convert one raw span (s_raw, e_raw) into pose indices.

    Inputs:
      s_raw, e_raw:
        raw indices from annotation file.
        If end_inclusive=True, e_raw is included.
        If end_inclusive=False, treat e_raw as already end-exclusive.

      frame_stride:
        If pose was extracted using every k-th raw frame, set k here.

    Mapping:
      if inclusive: e_raw_excl = e_raw + 1
      else:        e_raw_excl = e_raw

      s_pose = floor(s_raw / k)
      e_pose = ceil(e_raw_excl / k) = (e_raw_excl + k - 1) // k

    Why end uses ceil:
      If stride > 1, using floor shrinks spans too much and can kill positives.
    """
    k = max(1, int(frame_stride))

    s_raw = max(0, int(s_raw))
    e_raw = int(e_raw)

    e_excl = e_raw + 1 if end_inclusive else e_raw
    if e_excl <= s_raw:
        return None

    s_pose = s_raw // k
    e_pose = (e_excl + k - 1) // k  # ceil

    if clamp_len is not None:
        s_pose = max(0, min(s_pose, clamp_len))
        e_pose = max(0, min(e_pose, clamp_len))

    if e_pose <= s_pose:
        return None
    return [int(s_pose), int(e_pose)]


def merge_spans(spans: List[List[int]]) -> List[List[int]]:
    """
    Merge overlapping/adjacent spans.

    Input spans are [start, end_excl].
    Output is also [start, end_excl].
    """
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out: List[List[int]] = []
    for s, e in spans:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out


# -----------------------------
# 6) Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build LE2i labels + fall spans keyed by pose NPZ stems.")
    ap.add_argument("--raw_root", default="data/raw/LE2i", help="LE2i raw root containing scenes/*/Annotation_files")
    ap.add_argument("--npz_dir", default="data/interim/le2i/pose_npz", help="Pose NPZ dir for LE2i (canonical stems)")
    ap.add_argument("--out_labels", default="configs/labels/le2i.json")
    ap.add_argument("--out_spans", default="configs/labels/le2i_spans.json")

    ap.add_argument(
        "--include_unannotated_as_adl",
        action="store_true",
        help="If set, sequences with no annotation file are included as ADL (label=0). Otherwise excluded.",
    )
    ap.add_argument(
        "--end_inclusive",
        action="store_true",
        help="If set, treat annotation end index as inclusive (convert to end-exclusive by +1).",
    )
    ap.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="If pose extraction sampled every k-th raw frame, set k here (default: 1).",
    )
    ap.add_argument(
        "--clamp_to_npz_len",
        action="store_true",
        help="Clamp spans to the sequence length from NPZ (best-effort).",
    )
    args = ap.parse_args()

    stems = list_npz_stems(args.npz_dir)
    if not stems:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    labels: Dict[str, int] = {}
    spans_out: Dict[str, List[List[int]]] = {}

    scanned = 0
    included = 0
    fall_count = 0
    no_ann = 0

    for stem in stems:
        scanned += 1

        scene, vid = stem_to_scene_and_vid(stem)
        if scene is None or vid is None:
            # If we can't parse, skip. Better than writing wrong labels.
            continue

        ann = find_annotation_file(args.raw_root, scene, vid)
        if ann is None:
            no_ann += 1
            if args.include_unannotated_as_adl:
                labels[stem] = 0
                included += 1
            continue

        included += 1

        raw_spans = parse_span_txt(ann)

        clamp_len = read_seq_len(args.npz_dir, stem) if args.clamp_to_npz_len else None

        pose_spans: List[List[int]] = []
        for s_raw, e_raw in raw_spans:
            sp = raw_span_to_pose_span(
                s_raw,
                e_raw,
                end_inclusive=bool(args.end_inclusive),
                frame_stride=int(args.frame_stride),
                clamp_len=clamp_len,
            )
            if sp:
                pose_spans.append(sp)

        pose_spans = merge_spans(pose_spans)

        if pose_spans:
            labels[stem] = 1
            spans_out[stem] = pose_spans
            fall_count += 1
        else:
            labels[stem] = 0

    # Write outputs
    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)

    os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
    with open(args.out_spans, "w", encoding="utf-8") as f:
        json.dump(spans_out, f, indent=2, sort_keys=True)

    adl_count = sum(1 for v in labels.values() if v == 0)

    print(f"[OK] wrote labels → {args.out_labels} (total={len(labels)} fall={fall_count} adl={adl_count})")
    print(f"[OK] wrote spans  → {args.out_spans}  (videos_with_spans={len(spans_out)})")
    print(f"[info] stems scanned={scanned} included={included} no_annotation={no_ann}")
    if not args.include_unannotated_as_adl:
        print("[info] unannotated sequences were excluded (use --include_unannotated_as_adl to include as ADL).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
