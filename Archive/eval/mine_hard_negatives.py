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
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.ckpt import load_ckpt, get_cfg
from core.features import FeatCfg, read_window_npz, build_tcn_input, build_gcn_input
from core.models import build_model, pick_device, logits_1d


class WindowsForMining(Dataset):
    def __init__(self, windows_dir: str, feat_cfg: FeatCfg, fps_default: float, arch: str, two_stream: bool, neg_only: bool):
        self.root = windows_dir
        self.feat_cfg = feat_cfg
        self.fps_default = float(fps_default)
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)
        self.neg_only = bool(neg_only)

        self.files = sorted(glob.glob(os.path.join(self.root, "**", "*.npz"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .npz under: {self.root}")

        if self.neg_only:
            kept = []
            for p in self.files:
                _, _, _, _, _, meta = read_window_npz(p, fps_default=self.fps_default)
                if meta.y <= 0:  # 0 or -1 (unlabeled)
                    kept.append(p)
            self.files = kept
            if not self.files:
                raise RuntimeError(f"No negative/unlabeled windows found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        p = self.files[i]
        joints, motion, conf, mask, fps, _meta = read_window_npz(p, fps_default=self.fps_default)

        clip = (_meta.video_id or os.path.splitext(os.path.basename(p))[0])

        if self.arch == "tcn":
            X, _ = build_tcn_input(joints, motion, conf, mask, fps, self.feat_cfg)  # [T,C]
            return torch.from_numpy(X).float(), p, clip

        X, _ = build_gcn_input(joints, motion, conf, mask, fps, self.feat_cfg)  # [T,V,F]
        if self.two_stream:
            xy = X[..., 0:2]
            conf1 = X[..., -1:] if self.feat_cfg.use_conf_channel else None
            xj = np.concatenate([xy, conf1], axis=-1) if conf1 is not None else xy
            xm = X[..., 2:4] if self.feat_cfg.use_motion else np.zeros_like(xy, dtype=np.float32)
            return (torch.from_numpy(xj).float(), torch.from_numpy(xm).float()), p, clip

        return torch.from_numpy(X).float(), p, clip


def collate(batch):
    if isinstance(batch[0][0], tuple):
        xj, xm, ps, clips = [], [], [], []
        for (a, b), p, c in batch:
            xj.append(a); xm.append(b); ps.append(p); clips.append(c)
        return (torch.stack(xj, 0), torch.stack(xm, 0)), ps, clips
    xs, ps, clips = [], [], []
    for x, p, c in batch:
        xs.append(x); ps.append(p); clips.append(c)
    return torch.stack(xs, 0), ps, clips


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
    args = ap.parse_args()

    device = pick_device()

    bundle = load_ckpt(args.ckpt, map_location=device)
    arch, model_cfg, feat_cfg_d, data_cfg = get_cfg(bundle)

    feat_cfg = FeatCfg.from_dict(feat_cfg_d)
    fps_default = float(data_cfg.get("fps_default", 30.0))
    two_stream = bool(model_cfg.get("two_stream", False))

    ds = WindowsForMining(args.windows_dir, feat_cfg, fps_default, arch, two_stream=two_stream, neg_only=bool(args.neg_only))

    # infer input dims from one sample
    x0, _, _ = ds[0]
    if str(arch).lower() == "tcn":
        C = int(x0.shape[1])
        model = build_model("tcn", model_cfg, in_ch=C).to(device)
    else:
        if two_stream:
            xj0, xm0 = x0
            V = int(xj0.shape[1])
            Fj = int(xj0.shape[-1])
            Fm = int(xm0.shape[-1])
            model = build_model("gcn", model_cfg, num_joints=V, in_feats_j=Fj, in_feats_m=Fm).to(device)
        else:
            V = int(x0.shape[1])
            F = int(x0.shape[-1])
            model = build_model("gcn", model_cfg, num_joints=V, in_feats=F).to(device)

    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)
    rows_all: List[Tuple[str, str, float]] = []
    rows: List[Tuple[str, str, float]] = []
    for X, paths, clips in dl:
        if isinstance(X, tuple):
            X0, M0 = X
            # Two-stream model forward is model(xj, xm), not model((xj, xm)).
            logits = model(X0.to(device), M0.to(device))
        else:
            logits = model(X.to(device))
        log1d = logits_1d(logits)
        probs = torch.sigmoid(log1d).detach().cpu().numpy().reshape(-1)
        for p, c, pr in zip(paths, clips, probs):
            row = (p, c, float(pr))
            rows_all.append(row)
            if float(pr) >= float(args.min_p):
                rows.append(row)

    if not rows:
        print("[warn] No windows met min_p; falling back to top_k highest-scoring windows (ignoring min_p).")
        rows = rows_all

    # Sort by score, cap per clip
    rows.sort(key=lambda x: x[2], reverse=True)
    per_clip: Dict[str, int] = {}
    selected: List[Tuple[str, float]] = []
    for p, c, s in rows:
        k = per_clip.get(c, 0)
        if k >= int(args.max_per_clip):
            continue
        selected.append((p, s))
        per_clip[c] = k + 1
        if len(selected) >= int(args.top_k):
            break

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for p, _s in selected:
            f.write(p + "\n")

    print(f"[ok] wrote {len(selected)} hard negatives -> {args.out_txt}")


if __name__ == "__main__":
    main()
