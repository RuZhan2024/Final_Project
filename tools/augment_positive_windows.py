#!/usr/bin/env python3
"""Generate augmented positive windows for train-split NPZ datasets.

This script copies all windows from an input train directory into an output
train directory, then appends augmented copies for positive samples only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _augment_xy(xy: np.ndarray, rng: np.random.Generator, noise_std: float, scale_jitter: float) -> np.ndarray:
    out = xy.astype(np.float32, copy=True)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    out *= np.float32(scale)
    out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return np.clip(out, -2.0, 2.0)


def _augment_conf(conf: np.ndarray, rng: np.random.Generator, drop_prob: float) -> np.ndarray:
    out = conf.astype(np.float32, copy=True)
    if drop_prob > 0.0:
        drop = rng.random(size=out.shape) < drop_prob
        out[drop] *= 0.5
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Augment positive windows in train split.")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--copies_per_pos", type=int, default=1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--scale_jitter", type=float, default=0.04)
    p.add_argument("--conf_drop_prob", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=33724876)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(in_dir.glob("*.npz"))
    total = 0
    pos = 0
    aug = 0

    for src in files:
        total += 1
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())

        with np.load(src, allow_pickle=False) as z:
            y = int(z["y"]) if "y" in z.files else int(z["label"])
            if y != 1:
                continue
            pos += 1
            data = {k: z[k] for k in z.files}

        for i in range(args.copies_per_pos):
            d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
            d["xy"] = _augment_xy(d["xy"], rng, args.noise_std, args.scale_jitter)
            if "joints" in d:
                d["joints"] = d["xy"].copy()
            if "conf" in d:
                d["conf"] = _augment_conf(d["conf"], rng, args.conf_drop_prob)
            if "motion" in d:
                m = np.zeros_like(d["xy"], dtype=np.float32)
                m[1:] = d["xy"][1:] - d["xy"][:-1]
                d["motion"] = m
            if "seq_id" in d:
                d["seq_id"] = np.array(str(d["seq_id"]) + f"_aug{i+1}", dtype=d["seq_id"].dtype)
            if "src" in d:
                d["src"] = np.array(str(d["src"]) + f"|aug{i+1}", dtype=d["src"].dtype)
            out_name = src.stem + f"__aug{i+1}.npz"
            np.savez_compressed(out_dir / out_name, **d)
            aug += 1

    print(f"[ok] copied={total} positives={pos} aug_added={aug} out={out_dir}")


if __name__ == "__main__":
    main()
