#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
windows/make_fa_windows.py

Build "FA windows" directories from an existing windows root by keeping
only negative windows (y==0). Designed for --fa_dir in eval/fit_ops.py.

Default behavior:
- Input root has splits: train/val/test
- Output root has same splits
- Creates symlinks (no data duplication)

Optional:
- --only_neg_videos 1: keep only videos that contain NO positive windows
  in that split (requires video_id extraction from meta; best effort).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _iter_npz(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.npz")
    else:
        yield from root.glob("*.npz")


def _safe_int(x, default: int = -999) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_y_and_vid(fp: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Return (y, vid) where y in {-1,0,1} if possible.
    vid is best-effort (used only for --only_neg_videos).
    """
    try:
        with np.load(fp, allow_pickle=False) as z:
            # y: easiest case
            if "y" in z:
                y = _safe_int(np.asarray(z["y"]).reshape(-1)[0], default=None)
            elif "label" in z:
                y = _safe_int(np.asarray(z["label"]).reshape(-1)[0], default=None)
            else:
                y = None

            # best-effort video id
            vid = None
            for k in ("video_id", "seq_id", "seq_stem", "stem", "vid", "video", "video_name", "name"):
                if k in z.files:
                    raw = np.asarray(z[k]).reshape(-1)
                    if raw.size:
                        vid = str(raw[0])
                        if vid:
                            break

            if vid is None:
                # heuristic from filename
                stem = fp.stem
                # common patterns: <vid>__wXXXX or <vid>_sXXXX, etc.
                if "__" in stem:
                    vid = stem.split("__", 1)[0]
                else:
                    vid = stem.split("_w", 1)[0].split("_s", 1)[0]

            return y, vid
    except Exception:
        return None, None


def _link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_abs = src.resolve()

    if mode == "symlink":
        # Idempotent fast path: existing symlink already points to desired source.
        if dst.is_symlink():
            try:
                if dst.resolve() == src_abs:
                    return
            except Exception:
                pass
            dst.unlink()
        elif dst.exists():
            dst.unlink()

        try:
            dst.symlink_to(src_abs)
        except FileExistsError:
            # Handle races / stale state robustly.
            if dst.is_symlink():
                try:
                    if dst.resolve() == src_abs:
                        return
                except Exception:
                    pass
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src_abs)
    elif mode == "hardlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(src, dst)
    elif mode == "copy":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Input windows root (has train/val/test)")
    ap.add_argument("--out_root", required=True, help="Output FA windows root")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--recursive", type=int, default=1)
    ap.add_argument("--only_neg_videos", type=int, default=0)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    recursive = bool(int(args.recursive))
    only_neg_videos = bool(int(args.only_neg_videos))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    out_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        in_dir = in_root / split
        if not in_dir.exists():
            print(f"[warn] split dir missing, skip: {in_dir}")
            continue

        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        files = list(_iter_npz(in_dir, recursive=recursive))
        if not files:
            print(f"[warn] no npz files under: {in_dir}")
            continue

        # Pass 1: identify videos that have any positive window (if requested)
        vid_has_pos: Dict[str, bool] = {}
        if only_neg_videos:
            for fp in files:
                y, vid = _read_y_and_vid(fp)
                if vid is None:
                    continue
                if y == 1:
                    vid_has_pos[vid] = True

        kept = 0
        skipped_pos = 0
        skipped_vidpos = 0
        skipped_unknown = 0

        for fp in files:
            y, vid = _read_y_and_vid(fp)
            if y is None:
                skipped_unknown += 1
                continue
            if y == 1:
                skipped_pos += 1
                continue
            if y != 0:
                # unlabeled/unknown → skip for FA
                skipped_unknown += 1
                continue

            if only_neg_videos and vid is not None and vid_has_pos.get(vid, False):
                skipped_vidpos += 1
                continue

            dst = out_dir / fp.name
            _link(fp, dst, mode=args.mode)
            kept += 1

        print(
            f"[fa] {split:5s} in={len(files):5d} kept={kept:5d} "
            f"skip_pos={skipped_pos:5d} skip_vidpos={skipped_vidpos:5d} skip_unknown={skipped_unknown:5d} "
            f"mode={args.mode} only_neg_videos={int(only_neg_videos)}"
        )


if __name__ == "__main__":
    main()
