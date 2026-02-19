#!/usr/bin/env python3
"""
windows/make_windows.py  (redesigned + rewritten)

Goal
----
Turn cleaned pose sequences into fixed-length windows for training/eval, using:

- video-level labels: configs/labels/<ds>.json  {stem: "fall"/"adl" or 1/0}
- optional fall spans: configs/labels/<ds>_spans.json {stem: [[start, stop), ...]}

New pipeline concepts implemented
---------------------------------
1) Segment supervision (span-aware):
   - a window is POSITIVE only if it overlaps a fall span by enough frames.
2) Leak-safe split usage:
   - if train/val/test lists are provided, only stems in those lists are processed.
3) Balanced sampling per span:
   - keep up to K positives per span (evenly spread or random)
   - sample negatives with a target neg_ratio per fall video
   - optional "hard negatives" near the span boundaries
4) Quality gating:
   - reject windows with too many invalid joints / low confidence
   - uses 'mask' from preprocessed NPZ when available, else derives from conf+finite(xy).
5) Multi-stream ready windows:
   - always stores legacy keys: xy, conf, y
   - additionally stores: joints, motion, mask (uint8), valid_frac, overlap_frames, overlap_frac
6) Deterministic:
   - seed controls any randomness.

Input sequence NPZ schema (from preprocess_pose_npz.py)
------------------------------------------------------
Required:
  - xy   : (T, J, 2) float32
  - conf : (T, J) float32

Optional (if present, used):
  - mask       : (T, J) uint8/bool
  - frame_mask : (T,)   uint8/bool
  - fps        : scalar
  - seq_id/src/seq_stem

Output window NPZ schema
------------------------
Always:
  - xy, conf
  - joints     : same as xy (alias for model code)
  - motion     : (W, J, 2) float32, joints[t]-joints[t-1] (motion[0]=0)
  - mask       : (W, J) uint8
  - y          : int8 0/1
  - label      : int8 0/1 (backward compat)
  - fps        : float32
  - video_id, seq_id, src, seq_stem
  - w_start, w_end : int32, inclusive indices in original sequence
  - valid_frac : float32, mean(mask) within the window
  - overlap_frames : int16, max overlap with any span (0 if none)
  - overlap_frac   : float32, overlap_frames / W

Output layout
-------------
If split lists provided:
  out_dir/train/*.npz
  out_dir/val/*.npz
  out_dir/test/*.npz
Else:
  out_dir/unsplit/*.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np


# -------------------------
# Utilities
# -------------------------

_POS_STR = {"1", "true", "fall", "pos", "positive", "yes", "y", "t"}
_NEG_STR = {"0", "false", "adl", "neg", "negative", "no", "n", "f", "nonfall", "nofall", "normal"}


def to_binary_label(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return 1 if float(v) >= 0.5 else 0
    s = str(v).strip().lower()
    if s in _POS_STR:
        return 1
    if s in _NEG_STR:
        return 0
    raise ValueError(f"Unrecognized label value: {v!r}")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_stems_txt(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def index_sequences(npz_dir: str) -> Dict[str, str]:
    """
    Index all *.npz under npz_dir recursively: {stem: path}.

    If duplicate stems appear, the first encountered is kept and a warning is printed.
    """
    files = glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True)
    files.sort()
    idx: Dict[str, str] = {}
    dup = 0
    for p in files:
        stem = Path(p).stem
        if stem in idx:
            dup += 1
            continue
        idx[stem] = p
    if dup:
        print(f"[WARN] duplicate stems found in {npz_dir}: {dup} (kept first occurrence)")
    return idx


def as_py_str(x: Any) -> str:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return str(x.item())
            if x.size == 1:
                return str(x.reshape(-1)[0].item())
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)
    except Exception:
        return str(x)


def safe_fps(z: np.lib.npyio.NpzFile, fps_default: float) -> float:
    if "fps" in z.files:
        try:
            return float(np.array(z["fps"]).reshape(-1)[0])
        except Exception:
            return float(fps_default)
    return float(fps_default)


def normalize_spans(raw: Optional[dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Normalize spans to: {stem: [(start, stop), ...]} as ints.
    The span endpoints are *not* interpreted here; interpretation is handled later.
    """
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not raw:
        return out
    for k, v in raw.items():
        if not v:
            continue
        spans: List[Tuple[int, int]] = []
        for pair in v:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                continue
            spans.append((a, b))
        if spans:
            out[str(k)] = spans
    return out


def required_overlap_frames(W: int, min_overlap_frames: int, min_overlap_frac: float) -> int:
    if min_overlap_frames > 0:
        return max(1, int(min_overlap_frames))
    if min_overlap_frac > 0:
        return max(1, int(math.ceil(float(min_overlap_frac) * W)))
    return 1


def span_to_inclusive(span: Tuple[int, int], spans_end_exclusive: bool) -> Tuple[int, int]:
    """
    Convert a span to inclusive [start, end_inclusive].
    - if spans_end_exclusive: span is [start, stop) -> inclusive end = stop-1
    - else: span is [start, end_inclusive]
    """
    s, e = span
    return (s, e - 1) if spans_end_exclusive else (s, e)


def overlap_frames(w_start: int, w_end_incl: int, s_start: int, s_end_incl: int) -> int:
    lo = max(w_start, s_start)
    hi = min(w_end_incl, s_end_incl)
    return max(0, hi - lo + 1)


def max_overlap_with_spans(
    w_start: int,
    w_end_incl: int,
    spans: Sequence[Tuple[int, int]],
    spans_end_exclusive: bool,
) -> int:
    best = 0
    for sp in spans:
        ss, se = span_to_inclusive(sp, spans_end_exclusive)
        if se < ss:
            continue
        ov = overlap_frames(w_start, w_end_incl, ss, se)
        if ov > best:
            best = ov
    return best


def evenly_sample(items: List[int], k: int) -> List[int]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    xs = np.linspace(0, len(items) - 1, num=k)
    idx = sorted({int(round(x)) for x in xs})
    # pad if rounding reduced unique count
    if len(idx) < k:
        for i in range(len(items)):
            if i not in idx:
                idx.append(i)
                if len(idx) == k:
                    break
    return [items[i] for i in idx[:k]]


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


# -------------------------
# Core windowing
# -------------------------

def choose_balanced_windows_for_fall_video(
    starts: List[int],
    W: int,
    spans: Sequence[Tuple[int, int]],
    spans_end_exclusive: bool,
    req_ov: int,
    rng: np.random.Generator,
    pos_per_span: int,
    pos_pick_mode: str,
    neg_ratio: float,
    max_neg_per_video: int,
    hard_neg_margin: int,
    hard_neg_frac: float,
) -> List[Tuple[int, int, int]]:
    """
    For a fall video with spans:
      - label each start by overlap
      - select positives per span
      - select negatives with target ratio, optionally with hard negatives near spans

    Returns list of (start, y, ov_frames_max).
    """
    # Partition into pos/neg + remember max overlap
    pos_starts: List[int] = []
    neg_starts: List[int] = []
    ov_map: Dict[int, int] = {}
    for st in starts:
        ed = st + W - 1
        ov = max_overlap_with_spans(st, ed, spans, spans_end_exclusive)
        ov_map[st] = ov
        if ov >= req_ov:
            pos_starts.append(st)
        else:
            neg_starts.append(st)

    if not pos_starts:
        # no positives under overlap rule -> treat as no-span case at caller
        return []

    pos_pick_mode = pos_pick_mode.lower()
    if pos_pick_mode not in {"even", "random"}:
        pos_pick_mode = "even"

    # Per-span positive selection
    pos_picks: List[int] = []
    if pos_per_span > 0:
        for sp in spans:
            ss, se = span_to_inclusive(sp, spans_end_exclusive)
            if se < ss:
                continue
            hits = []
            for st in pos_starts:
                ed = st + W - 1
                if overlap_frames(st, ed, ss, se) >= req_ov:
                    hits.append(st)
            hits.sort()
            if not hits:
                continue
            if pos_pick_mode == "random":
                if len(hits) <= pos_per_span:
                    pick = hits
                else:
                    pick = rng.choice(hits, size=pos_per_span, replace=False).tolist()
                    pick.sort()
            else:
                pick = evenly_sample(hits, pos_per_span)
            pos_picks.extend(pick)
    else:
        pos_picks = list(pos_starts)

    # Unique while keeping order
    seen = set()
    pos_picks_u: List[int] = []
    for st in sorted(pos_picks):
        if st not in seen:
            seen.add(st)
            pos_picks_u.append(st)
    pos_picks = pos_picks_u

    # Negative sampling
    target_neg = int(math.ceil(max(1, len(pos_picks)) * float(neg_ratio)))
    target_neg = min(target_neg, int(max_neg_per_video))

    # Hard negatives: windows that are close to spans but not positive (ov < req_ov)
    hard: List[int] = []
    if hard_neg_margin > 0 and neg_starts:
        for st in neg_starts:
            ed = st + W - 1
            # if window intersects expanded span region (span +- margin), call it "hard"
            is_hard = False
            for sp in spans:
                ss, se = span_to_inclusive(sp, spans_end_exclusive)
                if se < ss:
                    continue
                ss2 = ss - hard_neg_margin
                se2 = se + hard_neg_margin
                if overlap_frames(st, ed, ss2, se2) > 0:
                    is_hard = True
                    break
            if is_hard:
                hard.append(st)

    hard = sorted(set(hard))
    easy = [st for st in neg_starts if st not in set(hard)]

    n_hard = int(round(target_neg * float(hard_neg_frac))) if hard else 0
    n_hard = min(n_hard, len(hard))
    n_easy = max(0, target_neg - n_hard)
    n_easy = min(n_easy, len(easy))

    neg_pick: List[int] = []
    if n_hard > 0:
        pick = hard if len(hard) <= n_hard else rng.choice(hard, size=n_hard, replace=False).tolist()
        neg_pick.extend(pick)
    if n_easy > 0:
        pick = easy if len(easy) <= n_easy else rng.choice(easy, size=n_easy, replace=False).tolist()
        neg_pick.extend(pick)

    neg_pick = sorted(set(neg_pick))

    out: List[Tuple[int, int, int]] = [(st, 1, ov_map.get(st, 0)) for st in pos_picks] + [(st, 0, ov_map.get(st, 0)) for st in neg_pick]
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--spans_json", default=None)
    ap.add_argument("--require_spans", type=int, default=0,
                    help="If 1: spans_json must exist and contain at least one span entry; abort otherwise.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--fps_default", type=float, default=30.0)

    # splits
    ap.add_argument("--train_list", default=None)
    ap.add_argument("--val_list", default=None)
    ap.add_argument("--test_list", default=None)

    # spans interpretation
    ap.add_argument("--spans_end_exclusive", action="store_true",
                    help="Interpret spans as [start, stop) (half-open).")
    ap.add_argument("--spans_end_inclusive", action="store_true",
                    help="Interpret spans as inclusive [start, end]. (Overrides exclusive).")

    ap.add_argument("--min_overlap_frames", type=int, default=0)
    ap.add_argument("--min_overlap_frac", type=float, default=0.25)

    # strategy
    ap.add_argument("--strategy", choices=["all", "balanced"], default="balanced")
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--pos_per_span", type=int, default=6)
    ap.add_argument("--pos_pick_mode", choices=["even", "random"], default="even")
    ap.add_argument("--neg_ratio", type=float, default=2.0)
    ap.add_argument("--max_neg_per_video", type=int, default=250)
    ap.add_argument("--max_windows_per_video_no_spans", type=int, default=250)
    ap.add_argument("--fallback_if_no_span", choices=["auto", "video_label", "all_neg", "skip_fall"], default="auto")

    # hard negatives
    ap.add_argument("--hard_neg_margin", type=int, default=12, help="Frames around span edges to mine hard negatives.")
    ap.add_argument("--hard_neg_frac", type=float, default=0.50, help="Fraction of negatives drawn from hard region.")

    # quality gate
    ap.add_argument("--conf_gate", type=float, default=0.20, help="Joint valid if conf>=conf_gate (used if mask missing).")
    ap.add_argument("--min_valid_frac", type=float, default=0.00, help="Require mean(mask) >= this (0 disables).")
    ap.add_argument("--min_avg_conf", type=float, default=0.00, help="Require mean(conf) >= this (0 disables).")
    ap.add_argument("--use_precomputed_mask", action="store_true",
                    help="If present, use 'mask' from preprocessed NPZ (recommended).")

    # output behaviour
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--write_manifest", action="store_true",
                    help="Write out_dir/<split>_manifest.jsonl with window path + y + meta.")

    args = ap.parse_args()

    if args.W <= 0 or args.stride <= 0:
        raise SystemExit("[ERR] W and stride must be positive.")

    if args.spans_end_inclusive and args.spans_end_exclusive:
        raise SystemExit("[ERR] Choose only one of --spans_end_exclusive / --spans_end_inclusive")

    # Default to half-open spans for the rewritten label scripts.
    spans_end_exclusive = True
    if args.spans_end_inclusive:
        spans_end_exclusive = False
    elif args.spans_end_exclusive:
        spans_end_exclusive = True

    req_ov = required_overlap_frames(args.W, args.min_overlap_frames, args.min_overlap_frac)

    labels_raw = load_json(args.labels_json)
    labels: Dict[str, int] = {str(k): to_binary_label(v) for k, v in labels_raw.items()}

    spans: Dict[str, List[Tuple[int, int]]] = {}
    if args.spans_json and os.path.exists(args.spans_json):
        spans = normalize_spans(load_json(args.spans_json))

    # Hard span enforcement (recommended for realistic event metrics)
    if int(getattr(args, 'require_spans', 0)) == 1:
        if not args.spans_json:
            raise SystemExit("[ERR] --require_spans=1 but --spans_json was not provided.")
        if not os.path.exists(args.spans_json):
            raise SystemExit(f"[ERR] --require_spans=1 but spans_json not found: {args.spans_json}")
        if not spans:
            raise SystemExit(f"[ERR] --require_spans=1 but spans_json is empty: {args.spans_json}")

    # Resolve auto fallback behaviour:
    # - If spans are provided (span-based datasets), default to skipping fall videos without spans (strict).
    # - If no spans are provided (video-label datasets), default to labeling windows by video label.
    if args.fallback_if_no_span == "auto":
        args.fallback_if_no_span = "skip_fall" if (args.spans_json and spans) else "video_label"
        print(f"[info] fallback_if_no_span=auto -> {args.fallback_if_no_span}")


    train_stems = set(read_stems_txt(args.train_list))
    val_stems = set(read_stems_txt(args.val_list))
    test_stems = set(read_stems_txt(args.test_list))
    use_splits = bool(train_stems or val_stems or test_stems)

    def split_for(stem: str) -> Optional[str]:
        if not use_splits:
            return "unsplit"
        if stem in train_stems:
            return "train"
        if stem in val_stems:
            return "val"
        if stem in test_stems:
            return "test"
        return None

    seq_index = index_sequences(args.npz_dir)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test", "unsplit"):
        (out_root / s).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # optional manifests
    manifests: Dict[str, List[dict]] = {"train": [], "val": [], "test": [], "unsplit": []}

    total = pos = neg = skipped_quality = missing_seq = 0

    # If splits provided, iterate stems from them; else iterate labels keys
    if use_splits:
        stems_ordered = []
        for s in (sorted(train_stems), sorted(val_stems), sorted(test_stems)):
            stems_ordered.extend(s)
    else:
        stems_ordered = sorted(labels.keys())

    for stem in stems_ordered:
        sp = split_for(stem)
        if sp is None:
            continue
        if stem not in labels:
            continue
        seq_path = seq_index.get(stem)
        if not seq_path:
            missing_seq += 1
            continue

        y_video = labels[stem]
        seq_spans = spans.get(stem, [])
        if y_video == 0:
            # Safety: ignore any spans for negative video label
            seq_spans = []

        with np.load(seq_path, allow_pickle=False) as z:
            if "xy" not in z.files or "conf" not in z.files:
                continue
            xy = np.nan_to_num(z["xy"]).astype(np.float32, copy=False)
            conf = np.nan_to_num(z["conf"]).astype(np.float32, copy=False)
            fps = safe_fps(z, args.fps_default)

            seq_id = as_py_str(z["seq_id"]) if "seq_id" in z.files else stem
            src = as_py_str(z["src"]) if "src" in z.files else ""
            seq_stem = as_py_str(z["seq_stem"]) if "seq_stem" in z.files else stem

            pre_mask = None
            if args.use_precomputed_mask and "mask" in z.files:
                pre_mask = z["mask"]

        T = int(xy.shape[0])
        if T < args.W:
            continue

        starts = list(range(0, T - args.W + 1, int(args.stride)))
        if not starts:
            continue

        chosen: List[Tuple[int, int, int]] = []  # (start, y, ov)

        if args.strategy == "all":
            if seq_spans:
                for st in starts:
                    ed = st + args.W - 1
                    ov = max_overlap_with_spans(st, ed, seq_spans, spans_end_exclusive)
                    y = 1 if ov >= req_ov else 0
                    chosen.append((st, y, ov))
            else:
                if y_video == 1 and args.fallback_if_no_span == "skip_fall":
                    chosen = []
                else:
                    y_const = 0 if (y_video == 1 and args.fallback_if_no_span == "all_neg") else y_video
                    chosen = [(st, y_const, 0) for st in starts]
        else:
            # balanced (default)
            if seq_spans:
                chosen = choose_balanced_windows_for_fall_video(
                    starts=starts,
                    W=args.W,
                    spans=seq_spans,
                    spans_end_exclusive=spans_end_exclusive,
                    req_ov=req_ov,
                    rng=rng,
                    pos_per_span=int(args.pos_per_span),
                    pos_pick_mode=str(args.pos_pick_mode),
                    neg_ratio=float(args.neg_ratio),
                    max_neg_per_video=int(args.max_neg_per_video),
                    hard_neg_margin=int(args.hard_neg_margin),
                    hard_neg_frac=float(args.hard_neg_frac),
                )
                if not chosen:
                    # no positives under overlap rule; fallback to no-span policy
                    if y_video == 1 and args.fallback_if_no_span == "skip_fall":
                        chosen = []
                    else:
                        y_const = 0 if (y_video == 1 and args.fallback_if_no_span == "all_neg") else y_video
                        cap = int(args.max_windows_per_video_no_spans)
                        if cap > 0 and len(starts) > cap:
                            starts2 = rng.choice(starts, size=cap, replace=False).tolist()
                            starts2.sort()
                        else:
                            starts2 = starts
                        chosen = [(st, y_const, 0) for st in starts2]
            else:
                # no spans
                if y_video == 1 and args.fallback_if_no_span == "skip_fall":
                    chosen = []
                else:
                    y_const = 0 if (y_video == 1 and args.fallback_if_no_span == "all_neg") else y_video
                    cap = int(args.max_windows_per_video_no_spans)
                    if cap > 0 and len(starts) > cap:
                        starts2 = rng.choice(starts, size=cap, replace=False).tolist()
                        starts2.sort()
                    else:
                        starts2 = starts
                    chosen = [(st, y_const, 0) for st in starts2]

        split_dir = out_root / sp

        for st, y, ov in chosen:
            ed = st + args.W - 1
            out_name = f"{stem}__w{st:06d}_{ed:06d}.npz"
            out_path = split_dir / out_name
            if args.skip_existing and out_path.exists():
                continue

            win_xy = xy[st: st + args.W]
            win_conf = conf[st: st + args.W]

            if pre_mask is not None:
                win_mask = np.array(pre_mask[st: st + args.W]).astype(bool, copy=False)
            else:
                win_mask = derive_mask(win_xy, win_conf, float(args.conf_gate))

            valid_frac = float(win_mask.mean()) if win_mask.size else 0.0
            if args.min_valid_frac > 0 and valid_frac < float(args.min_valid_frac):
                skipped_quality += 1
                continue
            if args.min_avg_conf > 0 and float(win_conf.mean()) < float(args.min_avg_conf):
                skipped_quality += 1
                continue

            joints = win_xy.astype(np.float32, copy=False)
            motion = compute_motion(joints)

            overlap_frac = float(ov) / float(args.W)

            np.savez_compressed(
                out_path,
                # legacy
                xy=joints,
                conf=win_conf.astype(np.float32, copy=False),
                y=np.int32(y),
                label=np.int32(y),

                # multi-stream ready
                joints=joints,
                motion=motion,
                mask=win_mask.astype(np.uint8),

                # overlap + quality
                valid_frac=np.float32(valid_frac),
                overlap_frames=np.int32(ov),
                overlap_frac=np.float32(overlap_frac),

                # meta
                fps=np.float32(fps),
                video_id=np.array(seq_id),
                seq_id=np.array(seq_id),
                src=np.array(src),
                seq_stem=np.array(seq_stem),
                w_start=np.int32(st),
                w_end=np.int32(ed),
            )

            total += 1
            if y == 1:
                pos += 1
            else:
                neg += 1

            if args.write_manifest:
                manifests[sp].append(
                    {
                        "path": str(out_path),
                        "stem": stem,
                        "y": int(y),
                        "w_start": int(st),
                        "w_end": int(ed),
                        "overlap_frames": int(ov),
                        "valid_frac": float(valid_frac),
                        "fps": float(fps),
                        "video_id": str(seq_id),
                    }
                )

    if args.write_manifest:
        for sp, rows in manifests.items():
            if not rows:
                continue
            mpath = out_root / f"{sp}_manifest.jsonl"
            with open(mpath, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] windows saved to: {args.out_dir}")
    print(f"[count] total={total} pos={pos} neg={neg} skipped_quality={skipped_quality} missing_seq={missing_seq}")

    if total > 0 and pos == 0:
        print("[warn] no POSITIVE windows were produced. If this dataset contains falls, check:")
        print('       - labels_json (does it contain any fall labels like 1 or "fall"?)')
        print("       - split lists (do they include fall videos?)")
        print("       - spans_json vs fall_class_id / overlap settings")
        print("       - for no-span datasets: use --fallback_if_no_span video_label")

    if total > 0 and neg == 0:
        print("[warn] no NEGATIVE windows were produced. Check labels_json / splits; you need ADL negatives for FA control.")
    if use_splits:
        print("[info] splits mode: enabled (processed only stems in split lists)")
    print(f"[info] spans_end_exclusive={spans_end_exclusive} req_overlap={req_ov} strategy={args.strategy}")


if __name__ == "__main__":
    main()
