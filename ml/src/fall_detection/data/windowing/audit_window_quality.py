#!/usr/bin/env python3
"""Audit pose/window quality for generated window datasets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _as_int(z: np.lib.npyio.NpzFile, key: str, default: int = 0) -> int:
    if key not in z.files:
        return int(default)
    try:
        return int(np.asarray(z[key]).reshape(-1)[0])
    except Exception:
        return int(default)


def _as_float(z: np.lib.npyio.NpzFile, key: str, default: float = math.nan) -> float:
    if key not in z.files:
        return float(default)
    try:
        return float(np.asarray(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def _as_str(z: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in z.files:
        return str(default)
    try:
        v = z[key].item() if np.ndim(z[key]) == 0 else z[key]
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return str(v)
    except Exception:
        return str(default)


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    xs = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if xs.size == 0:
        return {"min": math.nan, "p01": math.nan, "p05": math.nan, "p10": math.nan, "median": math.nan, "mean": math.nan, "p90": math.nan}
    qs = np.quantile(xs, [0.01, 0.05, 0.10, 0.50, 0.90])
    return {
        "min": float(np.min(xs)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p10": float(qs[2]),
        "median": float(qs[3]),
        "mean": float(np.mean(xs)),
        "p90": float(qs[4]),
    }


def _row_for_npz(path: str, split: str, root: str, min_valid_frac: float, min_frame_valid: float, min_avg_conf: float) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        conf = np.asarray(z["conf"], dtype=np.float32) if "conf" in z.files else np.empty((0, 0), dtype=np.float32)
        mask = np.asarray(z["mask"], dtype=bool) if "mask" in z.files else np.isfinite(conf)
        motion = np.asarray(z["motion"], dtype=np.float32) if "motion" in z.files else None

        if mask.size:
            frame_valid = mask.reshape(mask.shape[0], -1).mean(axis=1)
            valid_frac = _as_float(z, "valid_frac", float(mask.mean()))
        else:
            frame_valid = np.asarray([], dtype=np.float32)
            valid_frac = math.nan

        if conf.size:
            avg_conf = float(np.nanmean(conf))
            valid_conf = conf[mask] if mask.shape == conf.shape else conf.reshape(-1)
            valid_conf_mean = float(np.nanmean(valid_conf)) if valid_conf.size else math.nan
        else:
            avg_conf = math.nan
            valid_conf_mean = math.nan

        if motion is not None and motion.size:
            motion_mag = np.linalg.norm(motion, axis=-1)
            motion_mag_mean = float(np.nanmean(motion_mag))
            motion_mag_p95 = float(np.nanquantile(motion_mag, 0.95))
        else:
            motion_mag_mean = math.nan
            motion_mag_p95 = math.nan

        low_quality = (
            (np.isfinite(valid_frac) and valid_frac < min_valid_frac)
            or (np.isfinite(avg_conf) and avg_conf < min_avg_conf)
            or (frame_valid.size > 0 and float(np.min(frame_valid)) < min_frame_valid)
        )

        stem = Path(path).stem
        return {
            "split": split,
            "path": os.path.relpath(path, root),
            "stem": stem,
            "video_id": _as_str(z, "video_id", stem.split("__w", 1)[0]),
            "seq_stem": _as_str(z, "seq_stem", stem.split("__w", 1)[0]),
            "y": _as_int(z, "y", -999),
            "w_start": _as_int(z, "w_start", -1),
            "w_end": _as_int(z, "w_end", -1),
            "fps": _as_float(z, "fps", math.nan),
            "overlap_frames": _as_int(z, "overlap_frames", 0),
            "overlap_frac": _as_float(z, "overlap_frac", math.nan),
            "valid_frac": float(valid_frac),
            "avg_conf": float(avg_conf),
            "valid_conf_mean": float(valid_conf_mean),
            "frame_valid_min": float(np.min(frame_valid)) if frame_valid.size else math.nan,
            "frame_valid_p10": float(np.quantile(frame_valid, 0.10)) if frame_valid.size else math.nan,
            "low_frame_frac_25": float(np.mean(frame_valid < 0.25)) if frame_valid.size else math.nan,
            "low_frame_frac_50": float(np.mean(frame_valid < 0.50)) if frame_valid.size else math.nan,
            "zero_valid_frames": int(np.sum(frame_valid <= 0.0)) if frame_valid.size else 0,
            "motion_mag_mean": motion_mag_mean,
            "motion_mag_p95": motion_mag_p95,
            "low_quality": bool(low_quality),
        }


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        y_counts[str(r["y"])] += 1

    videos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        videos[str(r["video_id"])].append(r)

    worst_videos = []
    for vid, rs in videos.items():
        worst_videos.append(
            {
                "video_id": vid,
                "n_windows": len(rs),
                "n_pos": int(sum(int(r["y"]) == 1 for r in rs)),
                "valid_frac_mean": float(np.nanmean([float(r["valid_frac"]) for r in rs])),
                "valid_frac_min": float(np.nanmin([float(r["valid_frac"]) for r in rs])),
                "frame_valid_min": float(np.nanmin([float(r["frame_valid_min"]) for r in rs])),
                "low_quality_windows": int(sum(bool(r["low_quality"]) for r in rs)),
            }
        )
    worst_videos.sort(key=lambda r: (r["valid_frac_mean"], r["frame_valid_min"], -r["low_quality_windows"]))

    return {
        "n_windows": len(rows),
        "n_videos": len(videos),
        "y_counts": dict(sorted(y_counts.items())),
        "low_quality_windows": int(sum(bool(r["low_quality"]) for r in rows)),
        "valid_frac": _quantiles([float(r["valid_frac"]) for r in rows]),
        "avg_conf": _quantiles([float(r["avg_conf"]) for r in rows]),
        "frame_valid_min": _quantiles([float(r["frame_valid_min"]) for r in rows]),
        "low_frame_frac_50": _quantiles([float(r["low_frame_frac_50"]) for r in rows]),
        "worst_videos": worst_videos[:20],
    }


def _load_metric_details(items: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        per_video = blob.get("detail", {}).get("per_video", {})
        out[name] = {
            vid: {
                "recall": info.get("event_metrics", {}).get("recall", info.get("event_metrics", {}).get("event_recall")),
                "delay_s": info.get("event_metrics", {}).get("mean_delay_s"),
                "false_alerts": info.get("event_metrics", {}).get("n_false_alerts"),
                "alert_frac": info.get("state_counts", {}).get("alert_frac"),
            }
            for vid, info in per_video.items()
        }
    return out


def _write_video_csv(path: str, rows: Sequence[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    videos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        videos[str(r["video_id"])].append(r)

    out_rows: List[Dict[str, Any]] = []
    for vid, rs in sorted(videos.items()):
        out: Dict[str, Any] = {
            "video_id": vid,
            "splits": ",".join(sorted({str(r["split"]) for r in rs})),
            "n_windows": len(rs),
            "n_pos": int(sum(int(r["y"]) == 1 for r in rs)),
            "n_neg": int(sum(int(r["y"]) == 0 for r in rs)),
            "low_quality_windows": int(sum(bool(r["low_quality"]) for r in rs)),
            "valid_frac_mean": float(np.nanmean([float(r["valid_frac"]) for r in rs])),
            "valid_frac_min": float(np.nanmin([float(r["valid_frac"]) for r in rs])),
            "avg_conf_mean": float(np.nanmean([float(r["avg_conf"]) for r in rs])),
            "frame_valid_min": float(np.nanmin([float(r["frame_valid_min"]) for r in rs])),
            "low_frame_frac_50_mean": float(np.nanmean([float(r["low_frame_frac_50"]) for r in rs])),
        }
        for name, per_video in metrics.items():
            md = per_video.get(vid, {})
            out[f"{name}_recall"] = md.get("recall")
            out[f"{name}_delay_s"] = md.get("delay_s")
            out[f"{name}_false_alerts"] = md.get("false_alerts")
            out[f"{name}_alert_frac"] = md.get("alert_frac")
        out_rows.append(out)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Window root containing train/val/test splits.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_video_csv", default=None)
    ap.add_argument("--split", action="append", default=None, help="Split to scan. Repeatable. Default: train,val,test.")
    ap.add_argument("--metrics_json", action="append", default=[], help="Optional name=path metrics JSON to attach per-video details.")
    ap.add_argument("--min_valid_frac", type=float, default=0.50)
    ap.add_argument("--min_frame_valid", type=float, default=0.25)
    ap.add_argument("--min_avg_conf", type=float, default=0.15)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    splits = args.split or ["train", "val", "test"]
    rows: List[Dict[str, Any]] = []

    for split in splits:
        split_dir = os.path.join(root, split)
        for path in sorted(glob.glob(os.path.join(split_dir, "**", "*.npz"), recursive=True)):
            rows.append(_row_for_npz(path, split, root, args.min_valid_frac, args.min_frame_valid, args.min_avg_conf))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = _load_metric_details(args.metrics_json)
    if args.out_video_csv:
        _write_video_csv(args.out_video_csv, rows, metrics)
        print(f"[ok] wrote video csv: {args.out_video_csv}")

    by_split = {split: _summarize_rows([r for r in rows if r["split"] == split]) for split in splits}
    summary = {
        "root": root,
        "thresholds": {
            "min_valid_frac": float(args.min_valid_frac),
            "min_frame_valid": float(args.min_frame_valid),
            "min_avg_conf": float(args.min_avg_conf),
        },
        "total": _summarize_rows(rows),
        "splits": by_split,
        "metrics": metrics,
        "csv": args.out_csv,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote csv: {args.out_csv}")
    print(f"[ok] wrote json: {args.out_json}")


if __name__ == "__main__":
    main()
