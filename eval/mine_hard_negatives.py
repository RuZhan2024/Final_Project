#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/mine_hard_negatives.py

Mine hard negatives: windows that the model scores highly as positive within
a directory of *negative* windows (e.g., ADL-only clips, or val/test negatives).

Outputs a text file: one NPZ path per line, sorted by score descending.

Typical:
  python eval/mine_hard_negatives.py \
    --ckpt outputs/le2i_tcn_W48S12/best.pt \
    --windows_dir data/processed/le2i/windows_W48_S12/val \
    --neg_only 1 \
    --out_txt outputs/hardneg/le2i_tcn.txt \
    --top_k 5000 --max_per_clip 50 --min_p 0.50

Notes
- If --neg_only=1, only windows with y==0 OR y<0 are considered.
- Clip id is taken from window meta.video_id (fallback: filename). This works even when windows are stored flat.
- Output format (one NPZ path per line) matches models/train_tcn.py and models/train_gcn.py hard_neg_list loaders.
"""

from __future__ import annotations

# -------------------------
# Path bootstrap (so `from core.*` works when running as a script)
# -------------------------
import os as _os
import sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.ckpt import load_ckpt, get_cfg
from core.features import (
    FeatCfg,
    read_window_npz,
    build_tcn_input,
    build_canonical_input,
    split_gcn_two_stream,
)
from core.models import build_model, pick_device, logits_1d


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class WindowsForMining(Dataset):
    def __init__(
        self,
        windows_dir: str,
        feat_cfg: FeatCfg,
        fps_default: float,
        arch: str,
        two_stream: bool,
        neg_only: bool,
        verbose: bool = False,
    ):
        self.root = str(windows_dir)
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)
        self.neg_only = bool(neg_only)
        self.verbose = bool(verbose)

        self.files = sorted(glob.glob(os.path.join(self.root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {self.root}")

        if self.neg_only:
            kept: List[str] = []
            skipped_err = 0
            for p in self.files:
                try:
                    _j, _m, _c, _mask, _fps, meta = read_window_npz(p, fps_default=self.fps_default)
                except Exception as e:
                    skipped_err += 1
                    if self.verbose:
                        _warn(f"[warn] read_window_npz failed (skipping): {p}  err={type(e).__name__}: {e}")
                    continue
                if int(getattr(meta, "y", 0)) <= 0:  # 0 or -1 (unlabeled)
                    kept.append(p)

            self.files = kept
            if skipped_err and self.verbose:
                _warn(f"[info] skipped {skipped_err} windows due to read errors during neg_only filter.")
            if not self.files:
                raise RuntimeError(f"No negative/unlabeled windows found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        p = self.files[i]
        try:
            joints, motion, conf, mask, fps, meta = read_window_npz(p, fps_default=self.fps_default)
        except Exception as e:
            # Fail fast: mining on partially-read windows is worse than a crash.
            raise RuntimeError(f"read_window_npz failed for: {p}  err={type(e).__name__}: {e}") from e

        clip = (str(getattr(meta, "video_id", "")).strip() or os.path.splitext(os.path.basename(p))[0])

        # Always build the canonical representation first (same contract as train/eval scripts).
        Xc, _mask_used = build_canonical_input(
            joints,
            motion,
            conf,
            fps=float(fps),
            feat_cfg=self.feat_cfg,
            mask=mask,
        )  # [T,V,F]

        w_start = int(getattr(meta, "w_start", -1))
        w_end = int(getattr(meta, "w_end", -1))
        fps_v = float(fps) if fps and float(fps) > 0 else float(self.fps_default)

        if self.arch == "tcn":
            x = build_tcn_input(Xc, self.feat_cfg)  # [T, V*F]
            return torch.from_numpy(x).float(), p, clip, w_start, w_end, fps_v

        if self.two_stream:
            xj, xm = split_gcn_two_stream(Xc, self.feat_cfg)
            return (torch.from_numpy(xj).float(), torch.from_numpy(xm).float()), p, clip, w_start, w_end, fps_v

        return torch.from_numpy(Xc).float(), p, clip, w_start, w_end, fps_v


def collate(batch):
    # two-stream: ((xj,xm), path, clip, ws, we, fps)
    if isinstance(batch[0][0], tuple):
        xj, xm, ps, clips, ws, we, fps = [], [], [], [], [], [], []
        for (a, b), p, c, s, e, f in batch:
            xj.append(a)
            xm.append(b)
            ps.append(p)
            clips.append(c)
            ws.append(s)
            we.append(e)
            fps.append(f)
        return (torch.stack(xj, 0), torch.stack(xm, 0)), ps, clips, ws, we, fps

    xs, ps, clips, ws, we, fps = [], [], [], [], [], []
    for x, p, c, s, e, f in batch:
        xs.append(x)
        ps.append(p)
        clips.append(c)
        ws.append(s)
        we.append(e)
        fps.append(f)
    return torch.stack(xs, 0), ps, clips, ws, we, fps


def _dedup_keep(rows_sorted, dedup_shift_frames: int):
    """
    Keep at most one window within +/- dedup_shift_frames (by window center) per clip.

    This reduces "50 nearly-identical windows" from one false alarm region.
    """
    dedup_shift_frames = max(0, int(dedup_shift_frames))
    if dedup_shift_frames <= 0:
        return rows_sorted

    kept = []
    centers_by_clip: Dict[str, List[float]] = {}
    for p, clip, pr, ws, we, _fps in rows_sorted:
        c = 0.5 * (float(ws) + float(we)) if (ws >= 0 and we >= 0) else None
        if c is None:
            kept.append((p, clip, pr, ws, we, _fps))
            continue
        prev = centers_by_clip.setdefault(clip, [])
        if any(abs(c - pc) < dedup_shift_frames for pc in prev):
            continue
        prev.append(c)
        kept.append((p, clip, pr, ws, we, _fps))
    return kept


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min_p", type=float, default=0.20)
    ap.add_argument("--top_k", type=int, default=5000)
    ap.add_argument("--max_per_clip", type=int, default=50)
    ap.add_argument("--neg_only", type=int, default=1)
    ap.add_argument("--dedup_shift_frames", type=int, default=12,
                    help="Within each clip, suppress windows whose center is within this many frames of an already-kept window. Set 0 to disable.")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()

    bundle = load_ckpt(args.ckpt, map_location=device)
    arch, model_cfg, feat_cfg_d, data_cfg = get_cfg(bundle)

    feat_cfg = FeatCfg.from_dict(feat_cfg_d)
    fps_default = float(data_cfg.get("fps_default", 30.0))
    two_stream = bool(model_cfg.get("two_stream", False))

    ds = WindowsForMining(
        args.windows_dir,
        feat_cfg,
        fps_default,
        arch,
        two_stream=two_stream,
        neg_only=bool(args.neg_only),
        verbose=bool(args.verbose),
    )

    model = build_model(str(arch), model_cfg, feat_cfg, fps_default=fps_default).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)

    # rows_all: (path, clip, prob, ws, we, fps)
    rows_all: List[Tuple[str, str, float, int, int, float]] = []
    rows: List[Tuple[str, str, float, int, int, float]] = []

    for X, paths, clips, ws_list, we_list, fps_list in dl:
        if isinstance(X, tuple):
            X0, M0 = X
            logits = model(X0.to(device), M0.to(device))
        else:
            logits = model(X.to(device))

        probs = torch.sigmoid(logits_1d(logits)).detach().cpu().numpy().reshape(-1)

        for p, c, pr, ws, we, f in zip(paths, clips, probs, ws_list, we_list, fps_list):
            row = (str(p), str(c), float(pr), int(ws), int(we), float(f))
            rows_all.append(row)
            if float(pr) >= float(args.min_p):
                rows.append(row)

    if not rows:
        _warn("[warn] No windows met min_p; falling back to top_k highest-scoring windows (ignoring min_p).")
        rows = rows_all

    # Sort by prob desc
    rows.sort(key=lambda r: r[2], reverse=True)

    # De-duplicate near-identical windows within each clip
    rows = _dedup_keep(rows, dedup_shift_frames=int(args.dedup_shift_frames))

    # Cap per clip
    capped: List[Tuple[str, str, float, int, int, float]] = []
    per_clip: Dict[str, int] = {}
    for r in rows:
        clip = r[1]
        if per_clip.get(clip, 0) >= int(args.max_per_clip):
            continue
        per_clip[clip] = per_clip.get(clip, 0) + 1
        capped.append(r)
        if len(capped) >= int(args.top_k):
            break

    # Write NPZ paths only (train_* loaders expect this)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for p, _c, pr, _ws, _we, _fps in capped:
            f.write(f"{p}\n")

    print(f"[ok] wrote {len(capped)} hard negatives to: {args.out_txt}")
    if args.verbose:
        print(f"[info] unique clips={len(per_clip)}  dedup_shift_frames={int(args.dedup_shift_frames)}")


if __name__ == "__main__":
    main()
