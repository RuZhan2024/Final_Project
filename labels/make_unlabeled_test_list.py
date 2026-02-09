#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels/make_unlabeled_test_list.py

Utility: write a list of NPZ stems that match scene keywords
(typically used to create an "unlabeled test list" for LE2i).

Why this exists
---------------
In LE2i, some scenes may have no annotation files.
You often want to:
- exclude those from supervised training
- but keep them as an unlabeled "deployment-like" test set

This script selects stems based on scene name matching.

Key improvements vs older variants
----------------------------------
1) Matches scenes using the *stem scene token* (stem.split('__')[0]) rather than the full file path.
2) Optional: exclude stems that already exist in labels.json (useful when building an unlabeled list).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import List, Optional, Set


def norm(s: str) -> str:
    """Normalize strings so matching ignores case and space/underscore differences."""
    return re.sub(r"[\s_]+", "_", s.lower()).strip("_")


def list_npz_files(npz_dir: str) -> List[str]:
    """List all NPZ files under npz_dir (recursive)."""
    files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))
    # Ignore temp artifacts (e.g. "*.tmp.npz" or "*.npz.tmp.npz") that can appear
    # if a previous run was interrupted mid-write.
    files = [p for p in files if ".tmp" not in pathlib.Path(p).name]
    return files


def scene_from_stem(stem: str) -> str:
    """
    For your typical naming:
      Coffee_room_01__Videos__video__10_
    scene is the first token.
    """
    return stem.split("__")[0] if stem else ""


def load_labeled_stems(labels_json: Optional[str]) -> Set[str]:
    """
    Optional: load an existing labels JSON to exclude already-labeled stems.
    labels_json format can be:
      {stem: 0/1}  or  {stem: "fall"/"adl"} etc.
    We only care about keys.
    """
    if not labels_json:
        return set()
    if not os.path.exists(labels_json):
        return set()
    try:
        raw = json.loads(pathlib.Path(labels_json).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k) for k in raw.keys()}
    except Exception:
        return set()
    return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Root of pose NPZs (e.g., data/interim/le2i/pose_npz)")
    ap.add_argument("--out", required=True, help="Output txt with one stem per line")
    ap.add_argument("--scenes", nargs="+", required=True, help='Scene names to match, e.g. Office "Lecture room"')

    ap.add_argument(
        "--exclude_labels_json",
        default=None,
        help="If provided, exclude any stems already present in this labels JSON file.",
    )
    args = ap.parse_args()

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    # Normalize scene match keys
    wanted = {norm(s) for s in args.scenes}
    labeled = load_labeled_stems(args.exclude_labels_json)

    picked: List[str] = []
    for p in files:
        stem = pathlib.Path(p).stem
        if stem in labeled:
            continue

        scene = norm(scene_from_stem(stem))
        if scene in wanted:
            picked.append(stem)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(picked) + ("\n" if picked else ""))

    print(f"[OK] wrote {len(picked)} stems → {args.out}")
    if picked[:10]:
        print("[sample]", ", ".join(picked[:10]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
