#!/usr/bin/env python3
"""
pose/preprocess_pose_npz.py

Batch wrapper that preprocesses pose NPZ files BEFORE windowing.

This script intentionally contains NO preprocessing math.
It calls core.preprocess so offline preprocessing and server preprocessing match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from core.preprocess import OneEuroCfg, PreprocessCfg, preprocess_npz_file


def list_npz(in_dir: str, recursive: bool) -> list[Path]:
    root = Path(in_dir)
    return sorted(root.rglob("*.npz") if recursive else root.glob("*.npz"))


def parse_args():
    ap = argparse.ArgumentParser(description="Clean + normalise pose NPZs before windowing (core.preprocess).")

    # IO
    ap.add_argument("--in_dir", required=True, help="Directory containing pose NPZ files.")
    ap.add_argument("--out_dir", required=True, help="Where to write cleaned pose NPZ files.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip writing if output already exists.")

    # FPS
    ap.add_argument("--fps_default", type=float, default=30.0, help="Used if input NPZ has no fps.")
    ap.add_argument("--target_fps", type=float, default=30.0, help="Resample output to this fps (<=0 keeps src).")

    # Confidence gates (decoupled)
    ap.add_argument("--conf_gate", type=float, default=0.20, help="Final mask gate: joint valid if conf>=conf_gate.")
    ap.add_argument("--fill_conf_thr", type=float, default=0.20, help="Gap-fill eligibility threshold.")
    ap.add_argument("--norm_conf_gate", type=float, default=0.10, help="Normalization anchor gate (hips/shoulders).")

    # Gap fill
    ap.add_argument("--max_gap", type=int, default=4, help="Fill missing gaps up to this many frames (<=0 disables).")
    ap.add_argument("--fill_conf", choices=["keep", "thr", "min_neighbors", "linear"], default="thr")

    # Smoothing
    ap.add_argument("--one_euro", type=int, default=1, help="1=One-Euro smoothing, 0=disable One-Euro.")
    ap.add_argument("--one_euro_min_cutoff", type=float, default=1.0)
    ap.add_argument("--one_euro_beta", type=float, default=0.0)
    ap.add_argument("--one_euro_d_cutoff", type=float, default=1.0)

    ap.add_argument("--smooth_k", type=int, default=5, help="WMA window size used only if --one_euro 0.")

    # Normalization
    ap.add_argument("--normalize", choices=["none", "torso", "shoulder"], default="torso")
    ap.add_argument("--rotate", choices=["none", "shoulders"], default="none")
    ap.add_argument("--pelvis_fill", choices=["nearest", "zero"], default="nearest")

    # Frame quality gating
    ap.add_argument("--min_valid_ratio", type=float, default=0.25, help="Frame kept if mean(mask[t,:]) >= this.")
    ap.add_argument("--invalidate_bad_frames", action="store_true", help="NaN-out frames failing min_valid_ratio.")

    # Logging
    ap.add_argument("--log_every", type=int, default=200, help="Progress log frequency (0 disables).")

    return ap.parse_args()


def main():
    args = parse_args()

    files = list_npz(args.in_dir, args.recursive)
    if not files:
        raise SystemExit(f"[ERR] no .npz files under: {args.in_dir}")

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    norm_mode = str(args.normalize).lower()
    do_norm = norm_mode != "none"
    if norm_mode not in ("none", "torso", "shoulder"):
        raise SystemExit(f"[ERR] invalid --normalize: {args.normalize}")

    cfg = PreprocessCfg(
        # fps
        target_fps=float(args.target_fps) if float(args.target_fps) > 0 else float(args.fps_default),

        # gates
        conf_gate=float(args.conf_gate),
        fill_conf_thr=float(args.fill_conf_thr),
        norm_conf_gate=float(args.norm_conf_gate),

        # gap fill
        gap_fill_max=int(args.max_gap),
        gap_fill_conf=str(args.fill_conf),

        # normalization
        normalize=bool(do_norm),
        norm_mode=("torso" if norm_mode == "torso" else "shoulder"),
        pelvis_fallback=("carry" if str(args.pelvis_fill).lower() == "nearest" else "zero"),
        rotate_shoulders=(str(args.rotate).lower() == "shoulders"),

        # smoothing
        one_euro=bool(int(args.one_euro)),
        one_euro_cfg=OneEuroCfg(
            min_cutoff=float(args.one_euro_min_cutoff),
            beta=float(args.one_euro_beta),
            d_cutoff=float(args.one_euro_d_cutoff),
        ),
        wma_k=int(args.smooth_k),
    )

    ok, fail, skipped = 0, 0, 0

    for i, p in enumerate(files, start=1):
        rel = p.resolve().relative_to(in_root)
        out_p = out_root / rel

        if args.skip_existing and out_p.exists():
            skipped += 1
        else:
            try:
                preprocess_npz_file(
                    str(p),
                    str(out_p),
                    cfg,
                    fps_default=float(args.fps_default),
                    frame_gate=float(args.min_valid_ratio),
                    zero_bad_frames=bool(args.invalidate_bad_frames),
                )
                ok += 1
            except Exception as e:
                print(f"[ERR] {p}: {e}")
                fail += 1

        if int(args.log_every) > 0 and (i % int(args.log_every) == 0):
            print(f"[prog] {i}/{len(files)}  ok={ok}  fail={fail}  skipped={skipped}")

    print(f"[done] wrote {ok} files to {out_root}  (fail={fail}, skipped={skipped})")


if __name__ == "__main__":
    main()
