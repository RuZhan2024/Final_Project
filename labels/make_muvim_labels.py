
"""
MUVIM label + fall-span builder.

Produces:
  - video-level labels JSON: stem -> "adl"/"fall"
  - optional spans JSON:     stem -> [[start, stop], ...]  (half-open: [start, stop))

Why spans?
----------
If you label an entire "FallXX" sequence as positive, many pre-fall / post-fall
windows become positive too. That inflates positives after windowing and can
push thresholds very low and/or create many false alarms.

This script can ingest ZED_RGB.csv (frame-level fall annotations) so you can
label windows as positive only when they overlap the annotated fall intervals.

Assumptions
-----------
- NPZ stems (or their parent folder names) contain "Fall<video_id>" or "NonFall<video_id>"
  e.g., Fall0, Fall10, NonFall7, etc.
- ZED_RGB.csv uses columns: Participant, Trial, Video, Start, Stop, ToD, FallLength
  where FallLength ~= Stop - Start, so we treat spans as half-open [Start, Stop).
"""

import os
import re
import glob
import json
import pathlib
import argparse
import csv
from collections import defaultdict
from typing import Optional


_FALL_ID_RE = re.compile(r"(?:^|[^a-z0-9])fall\s*0*([0-9]+)", re.IGNORECASE)
_VIDEO_ID_RE = re.compile(r"(?:^|[^a-z0-9])video\s*0*([0-9]+)", re.IGNORECASE)


def infer_label_from_path(npz_path: str) -> str:
    """Infer video-level label from filename/parent folder conventions."""
    p = pathlib.Path(npz_path)
    stem = p.stem.lower()
    parent = (p.parent.name.lower() if p.parent is not None else "")

    if stem.startswith("nonfall") or parent.startswith("nonfall"):
        return "adl"
    if stem.startswith("fall") or parent.startswith("fall"):
        return "fall"

    for part in (x.lower() for x in p.parts):
        if part == "fall":
            return "fall"
        if part == "adl" or part == "nonfall":
            return "adl"

    return "adl"


def extract_video_id_from_stem(stem: str) -> Optional[int]:
    """
    Extract numeric id for a fall video.
    Prefer patterns like 'Fall10'. Also accept 'Video10' if present.

    Note: regex avoids matching "NonFall10" because the preceding char is alphanumeric.
    """
    s = str(stem)
    m = _FALL_ID_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = _VIDEO_ID_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def load_muvim_spans_from_csv(csv_path: str) -> dict[int, list[list[int]]]:
    """
    Return mapping: video_id -> list of [start, stop] (half-open, stop exclusive).
    """
    spans_by_vid = defaultdict(list)

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
                s = int(row["Start"])
                e = int(row["Stop"])
            except Exception:
                continue

            if s < 0:
                s = 0
            if e <= s:
                continue  # ignore zero/negative spans

            spans_by_vid[vid].append([s, e])

    for vid in spans_by_vid:
        spans_by_vid[vid].sort(key=lambda x: (x[0], x[1]))

    return dict(spans_by_vid)


def apply_padding(spans: list[list[int]], pad_pre: int, pad_post: int) -> list[list[int]]:
    """Pad half-open spans [s,e) to [max(0,s-pad_pre), e+pad_post)."""
    out = []
    for s, e in spans:
        s2 = max(0, int(s) - int(pad_pre))
        e2 = int(e) + int(pad_post)
        if e2 > s2:
            out.append([s2, e2])
    return out


def main():
    ap = argparse.ArgumentParser(description="Build MUVIM labels and optional fall spans.")
    ap.add_argument(
        "--npz_dir",
        default="data/interim/muvim/pose_npz",
        help="MUVIM pose NPZs from extract_2d_from_images.py",
    )
    ap.add_argument(
        "--out_labels",
        default="configs/labels/muvim.json",
        help="Where to write video-level labels JSON.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="(deprecated) same as --out_labels",
    )
    ap.add_argument(
        "--zed_csv",
        default=None,
        help="Path to ZED_RGB.csv (frame-level fall intervals). If provided, also writes spans JSON.",
    )
    ap.add_argument(
        "--out_spans",
        default="configs/labels/muvim_spans.json",
        help="Where to write spans JSON (stem -> [[start,stop],...], half-open).",
    )
    ap.add_argument("--pad_pre", type=int, default=0, help="Pad span starts by this many frames.")
    ap.add_argument("--pad_post", type=int, default=0, help="Pad span ends by this many frames.")

    args = ap.parse_args()
    if args.out is not None:
        args.out_labels = args.out

    files = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    # 1) video-level labels
    labels = {}
    stems = []
    for p in files:
        stem = pathlib.Path(p).stem
        stems.append(stem)
        labels[stem] = infer_label_from_path(p)

    # 2) optional spans
    spans: dict[str, list[list[int]]] = {}
    if args.zed_csv:
        spans_by_vid = load_muvim_spans_from_csv(args.zed_csv)

        matched = 0
        for stem in stems:
            vid = extract_video_id_from_stem(stem)
            if vid is None:
                continue
            if vid in spans_by_vid:
                sps = apply_padding(spans_by_vid[vid], args.pad_pre, args.pad_post)
                if sps:
                    spans[stem] = sps
                    matched += 1
                    labels[stem] = "fall"  # spans imply this is a fall video

        print(f"[info] loaded spans: {len(spans_by_vid)} unique video IDs from {args.zed_csv}")
        print(f"[info] stems matched to spans: {matched} (out of {len(stems)})")

    # write outputs
    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[OK] wrote {len(labels)} labels → {args.out_labels} "
          f"(fall={sum(v=='fall' for v in labels.values())}, adl={sum(v=='adl' for v in labels.values())})")

    if args.zed_csv:
        os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
        with open(args.out_spans, "w") as f:
            json.dump(spans, f, indent=2)
        print(f"[OK] wrote {len(spans)} span entries → {args.out_spans} (half-open [start, stop))")


if __name__ == "__main__":
    main()
