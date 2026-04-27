#!/usr/bin/env python3
"""
windows/make_unlabeled_windows.py  (rewritten)

Create fixed-length windows for unlabeled sequences (e.g., LE2i Office/Lecture).

This version matches the redesigned make_windows.py schema:
  - saves xy/conf + joints/motion/mask
  - y = -1, label = -1
  - supports quality gating using precomputed sequence mask if present

Output:
  {out_dir}/{subset}/{stem}__w{start:06d}_{end:06d}.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import random
import sys
from typing import Dict, List, Any

import numpy as np

# Support running directly without pre-set PYTHONPATH.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from fall_detection.data.adapters import build_adapter
except Exception:
    build_adapter = None

def read_stems(stems_txt: str) -> List[str]:
    out: List[str] = []
    with open(stems_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def index_npz(npz_root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    files = glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True)
    for p in files:
        idx[pathlib.Path(p).stem] = p
    return idx


def safe_fps(z, fps_default: float) -> float:
    if "fps" in z.files:
        try:
            return float(np.array(z["fps"]).reshape(-1)[0])
        except Exception:
            return float(fps_default)
    return float(fps_default)


def derive_mask(xy: np.ndarray, conf: np.ndarray, conf_gate: float) -> np.ndarray:
    finite = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    if conf_gate > 0:
        return (conf >= conf_gate) & finite
    return finite


def compute_motion(joints: np.ndarray) -> np.ndarray:
    motion = np.zeros_like(joints, dtype=np.float32)
    motion[1:] = joints[1:] - joints[:-1]
    motion[0] = 0.0
    return motion


def window_passes_quality(mask_w: np.ndarray, conf_w: np.ndarray, min_valid_frac: float, min_avg_conf: float) -> bool:
    if min_avg_conf > 0 and float(conf_w.mean()) < float(min_avg_conf):
        return False
    if min_valid_frac > 0 and float(mask_w.mean()) < float(min_valid_frac):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stems_txt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument(
        "--adapter_dataset",
        default="",
        help="Optional dataset adapter mode: le2i|caucafall|muvim|urfall|urfd.",
    )
    ap.add_argument(
        "--adapter_urfall_target_fps",
        type=float,
        default=25.0,
        help="Target FPS used by URFall adapter temporal resampling.",
    )
    ap.add_argument("--subset", default="test_unlabeled")
    ap.add_argument("--seed", type=int, default=33724876)

    ap.add_argument("--max_windows_per_video", type=int, default=400, help="Cap windows per video (0 disables).")

    # quality gating
    ap.add_argument("--use_precomputed_mask", action="store_true")
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--min_valid_frac", type=float, default=0.00)
    ap.add_argument("--min_avg_conf", type=float, default=0.00)

    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    if args.W <= 0 or args.stride <= 0:
        raise SystemExit("[ERR] W and stride must be positive integers.")

    adapter = None
    if str(args.adapter_dataset).strip():
        if build_adapter is None:
            raise SystemExit("[ERR] Adapter requested but fall_detection.data.adapters is unavailable.")
        adapter = build_adapter(
            str(args.adapter_dataset).strip().lower(),
            urfall_target_fps=float(args.adapter_urfall_target_fps),
        )
        print(f"[info] adapter enabled: dataset={adapter.dataset_name}")

    rng = random.Random(int(args.seed))

    stems = read_stems(args.stems_txt)
    if not stems:
        raise SystemExit(f"[ERR] No stems found in {args.stems_txt}")

    idx = index_npz(args.npz_dir)

    out_base = os.path.join(args.out_dir, args.subset)
    os.makedirs(out_base, exist_ok=True)

    total_win = 0
    missing: List[str] = []
    too_short = 0
    skipped_quality = 0

    for stem in stems:
        p = idx.get(stem)
        if p is None:
            missing.append(stem)
            continue

        if adapter is not None:
            seq = adapter.load_sequence(p, fps_default=float(args.fps_default))
            xy = np.nan_to_num(seq.joints_xy).astype(np.float32, copy=False)
            if seq.conf is not None:
                conf = np.nan_to_num(seq.conf).astype(np.float32, copy=False)
            elif seq.mask is not None:
                conf = np.asarray(seq.mask, dtype=np.float32)
            else:
                conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
            fps = float(seq.fps)
            seq_id = str(seq.meta.get("seq_id", seq.meta.get("video_id", stem)))
            src = str(seq.meta.get("src", ""))
            seq_stem = str(seq.meta.get("seq_stem", stem))
            pre_mask = np.asarray(seq.mask, dtype=bool) if (args.use_precomputed_mask and seq.mask is not None) else None
        else:
            with np.load(p, allow_pickle=False) as z:
                if "xy" not in z.files or "conf" not in z.files:
                    continue
                xy = np.nan_to_num(z["xy"]).astype(np.float32, copy=False)
                conf = np.nan_to_num(z["conf"]).astype(np.float32, copy=False)
                fps = safe_fps(z, args.fps_default)

                seq_id = str(stem)
                src = ""
                seq_stem = str(stem)
                if "seq_id" in z.files:
                    seq_id = str(np.array(z["seq_id"]).reshape(-1)[0])
                if "src" in z.files:
                    src = str(np.array(z["src"]).reshape(-1)[0])
                if "seq_stem" in z.files:
                    seq_stem = str(np.array(z["seq_stem"]).reshape(-1)[0])

                pre_mask = None
                if args.use_precomputed_mask and "mask" in z.files:
                    pre_mask = z["mask"]

        T = int(xy.shape[0])
        if T < args.W:
            too_short += 1
            continue

        starts = list(range(0, T - args.W + 1, int(args.stride)))
        if args.max_windows_per_video > 0 and len(starts) > args.max_windows_per_video:
            starts = rng.sample(starts, args.max_windows_per_video)
            starts = sorted(starts)

        for st in starts:
            ed = st + args.W - 1
            out_name = f"{stem}__w{st:06d}_{ed:06d}.npz"
            out_path = os.path.join(out_base, out_name)
            if args.skip_existing and os.path.exists(out_path):
                continue

            xy_w = xy[st:ed + 1]
            conf_w = conf[st:ed + 1]

            if pre_mask is not None:
                mask_w = np.array(pre_mask[st:ed + 1]).astype(bool, copy=False)
            else:
                mask_w = derive_mask(xy_w, conf_w, float(args.conf_gate))

            if not window_passes_quality(mask_w, conf_w, float(args.min_valid_frac), float(args.min_avg_conf)):
                skipped_quality += 1
                continue

            joints = xy_w.astype(np.float32, copy=False)
            motion = compute_motion(joints)
            valid_frac = float(mask_w.mean()) if mask_w.size else 0.0

            np.savez_compressed(
                out_path,
                xy=joints,
                conf=conf_w,
                y=np.int64(-1),
                label=np.int64(-1),
                joints=joints,
                motion=motion,
                mask=mask_w.astype(np.uint8),
                valid_frac=np.float32(valid_frac),
                overlap_frames=np.int16(0),
                overlap_frac=np.float32(0.0),
                fps=np.float32(fps),
                video_id=np.array(seq_id),
                seq_id=np.array(seq_id),
                src=np.array(src),
                seq_stem=np.array(seq_stem),
                w_start=np.int32(st),
                w_end=np.int32(ed),
            )
            total_win += 1

    print(f"[OK] wrote {total_win} unlabeled windows → {out_base}")
    if too_short:
        print(f"[WARN] {too_short} sequences shorter than W={args.W} (skipped).")
    if skipped_quality:
        print(f"[WARN] skipped_quality={skipped_quality}")
    if missing:
        print(f"[WARN] {len(missing)} stems not found in {args.npz_dir}. Examples: {missing[:5]}")


if __name__ == "__main__":
    main()
