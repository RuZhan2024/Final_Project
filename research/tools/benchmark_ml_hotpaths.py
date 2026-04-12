#!/usr/bin/env python3
"""Benchmark key ML/runtime hot paths for on-device fall detection."""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict
import statistics

import numpy as np

# Allow running directly without pre-setting PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.alerting import AlertCfg, sweep_alert_policy_from_windows
from core.confirm import confirm_scores_window
from core.features import FeatCfg, build_canonical_input
from pose.preprocess_pose_npz import linear_fill_small_gaps, smooth_weighted_moving_average
from server.routes.monitor import _resample_pose_window


@dataclass
class BenchResult:
    name: str
    iters: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _run_bench(name: str, fn: Callable[[], Any], *, iters: int, warmup: int) -> BenchResult:
    for _ in range(max(0, warmup)):
        fn()

    samples = np.empty((iters,), dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0

    return BenchResult(
        name=name,
        iters=int(iters),
        warmup=int(warmup),
        mean_ms=float(np.mean(samples)),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
    )


def _make_pose_data(T: int, J: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.5, 0.25, size=(T, J, 2)).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, size=(T, J)).astype(np.float32)

    # Simulate noisy extraction: missing joints/occlusions.
    miss = rng.uniform(0.0, 1.0, size=(T, J)) < 0.08
    xy[miss] = np.nan
    conf[miss] = 0.0
    return xy, conf


def _make_alert_data(n_videos: int, windows_per_video: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = n_videos * windows_per_video

    probs = rng.uniform(0.0, 1.0, size=(N,)).astype(np.float32)
    y_true = (rng.uniform(0.0, 1.0, size=(N,)) < 0.08).astype(np.int32)
    video_ids = np.repeat(np.array([f"v{i:04d}" for i in range(n_videos)]), windows_per_video)

    stride = 12
    wlen = 48
    ws = np.tile(np.arange(0, windows_per_video * stride, stride, dtype=np.int32), n_videos)
    we = ws + (wlen - 1)
    fps = np.full((N,), 25.0, dtype=np.float32)

    return {
        "probs": probs,
        "y_true": y_true,
        "video_ids": video_ids,
        "w_start": ws,
        "w_end": we,
        "fps": fps,
    }


def _make_stream_payload(n_src: int, j: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    t_ms = np.cumsum(rng.uniform(28.0, 40.0, size=(n_src,))).astype(np.float64)
    xy = rng.uniform(0.0, 1.0, size=(n_src, j, 2)).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, size=(n_src, j)).astype(np.float32)
    return {
        "raw_t_ms": t_ms.tolist(),
        "raw_xy": xy.tolist(),
        "raw_conf": conf.tolist(),
        "raw_xy_flat": xy.reshape(-1).astype(np.float32, copy=False).tolist(),
        "raw_conf_flat": conf.reshape(-1).astype(np.float32, copy=False).tolist(),
        "raw_joints": int(j),
        "target_fps": 25.0,
        "target_T": 48,
        "window_end_t_ms": float(t_ms[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Safe Guard ML hot paths")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pose_T", type=int, default=1800)
    ap.add_argument("--pose_J", type=int, default=33)
    ap.add_argument("--videos", type=int, default=30)
    ap.add_argument("--windows_per_video", type=int, default=250)
    ap.add_argument("--stream_src_frames", type=int, default=80)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--repeats", type=int, default=1, help="Run full benchmark suite this many times and aggregate.")
    ap.add_argument("--vary_seed", action="store_true", help="Vary synthetic data seed across repeats.")
    args = ap.parse_args()
    repeats = max(1, int(args.repeats))
    all_results = []
    for rep in range(repeats):
        seed = int(args.seed + rep * 17) if args.vary_seed else int(args.seed)
        xy, conf = _make_pose_data(args.pose_T, args.pose_J, seed)
        alert_data = _make_alert_data(args.videos, args.windows_per_video, seed + 1)
        stream_payload = _make_stream_payload(args.stream_src_frames, args.pose_J, seed + 2)

        alert_cfg = AlertCfg(
            ema_alpha=0.2,
            k=2,
            n=3,
            tau_high=0.85,
            tau_low=0.68,
            cooldown_s=30.0,
            confirm=False,
        )
        rep_results = []
        rep_results.append(
            _run_bench(
                "pose.smooth_weighted_moving_average",
                lambda: smooth_weighted_moving_average(xy, conf, conf_thr=0.2, k=5),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        mask = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1]) & (conf >= 0.2)
        rep_results.append(
            _run_bench(
                "confirm.confirm_scores_window",
                lambda: confirm_scores_window(xy, mask, 25.0, tail_s=1.0, smooth="median"),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        rep_results.append(
            _run_bench(
                "pose.linear_fill_small_gaps",
                lambda: linear_fill_small_gaps(xy, conf, conf_thr=0.2, max_gap=4, fill_conf="thr"),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        cfg_feat = FeatCfg(
            center="pelvis",
            use_motion=True,
            use_bone=True,
            use_bone_length=True,
            use_conf_channel=True,
            motion_scale_by_fps=True,
            conf_gate=0.2,
            use_precomputed_mask=True,
        )
        rep_results.append(
            _run_bench(
                "features.build_canonical_input",
                lambda: build_canonical_input(
                    joints_xy=xy,
                    motion_xy=None,
                    conf=conf,
                    mask=mask,
                    fps=25.0,
                    feat_cfg=cfg_feat,
                ),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        rep_results.append(
            _run_bench(
                "alerting.sweep_alert_policy_from_windows",
                lambda: sweep_alert_policy_from_windows(
                    alert_data["probs"],
                    alert_data["y_true"],
                    alert_data["video_ids"],
                    alert_data["w_start"],
                    alert_data["w_end"],
                    alert_data["fps"],
                    alert_base=alert_cfg,
                    thr_min=0.10,
                    thr_max=0.90,
                    thr_step=0.05,
                    tau_low_ratio=0.8,
                    time_mode="center",
                ),
                iters=args.iters,
                warmup=max(1, args.warmup // 2),
            )
        )
        # Confirm-enabled policy benchmark (uncertainty-aware deployment mode).
        alert_cfg_confirm = AlertCfg(
            ema_alpha=0.2,
            k=2,
            n=3,
            tau_high=0.85,
            tau_low=0.68,
            cooldown_s=30.0,
            confirm=True,
            confirm_s=2.0,
            confirm_min_lying=0.65,
            confirm_max_motion=0.08,
            confirm_require_low=True,
        )
        lying_score = np.clip(0.6 + 0.35 * alert_data["probs"], 0.0, 1.0).astype(np.float32, copy=False)
        motion_score = np.clip(0.25 - 0.2 * alert_data["probs"], 0.0, 1.0).astype(np.float32, copy=False)
        rep_results.append(
            _run_bench(
                "alerting.sweep_alert_policy_from_windows(confirm)",
                lambda: sweep_alert_policy_from_windows(
                    alert_data["probs"],
                    alert_data["y_true"],
                    alert_data["video_ids"],
                    alert_data["w_start"],
                    alert_data["w_end"],
                    alert_data["fps"],
                    alert_base=alert_cfg_confirm,
                    thr_min=0.10,
                    thr_max=0.90,
                    thr_step=0.05,
                    tau_low_ratio=0.8,
                    time_mode="center",
                    lying_score=lying_score,
                    motion_score=motion_score,
                ),
                iters=args.iters,
                warmup=max(1, args.warmup // 2),
            )
        )
        rep_results.append(
            _run_bench(
                "server.monitor._resample_pose_window(nested)",
                lambda: _resample_pose_window(
                    raw_t_ms=stream_payload["raw_t_ms"],
                    raw_xy=stream_payload["raw_xy"],
                    raw_conf=stream_payload["raw_conf"],
                    target_fps=stream_payload["target_fps"],
                    target_T=stream_payload["target_T"],
                    window_end_t_ms=stream_payload["window_end_t_ms"],
                ),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        rep_results.append(
            _run_bench(
                "server.monitor._resample_pose_window(flat)",
                lambda: _resample_pose_window(
                    raw_t_ms=stream_payload["raw_t_ms"],
                    raw_xy=[],
                    raw_conf=None,
                    raw_xy_flat=stream_payload["raw_xy_flat"],
                    raw_conf_flat=stream_payload["raw_conf_flat"],
                    raw_joints=stream_payload["raw_joints"],
                    target_fps=stream_payload["target_fps"],
                    target_T=stream_payload["target_T"],
                    window_end_t_ms=stream_payload["window_end_t_ms"],
                ),
                iters=args.iters,
                warmup=args.warmup,
            )
        )
        all_results.append(rep_results)

    aggregate = []
    if repeats > 1:
        by_name = {}
        for rep in all_results:
            for r in rep:
                by_name.setdefault(r.name, []).append(r)
        for name in sorted(by_name.keys()):
            rows = by_name[name]
            p50_vals = [float(x.p50_ms) for x in rows]
            mean_vals = [float(x.mean_ms) for x in rows]
            p95_vals = [float(x.p95_ms) for x in rows]
            aggregate.append(
                {
                    "name": name,
                    "runs": int(len(rows)),
                    "p50_ms_median": float(statistics.median(p50_vals)),
                    "p50_ms_mean": float(statistics.mean(p50_vals)),
                    "p95_ms_median": float(statistics.median(p95_vals)),
                    "p95_ms_mean": float(statistics.mean(p95_vals)),
                    "mean_ms_median": float(statistics.median(mean_vals)),
                    "mean_ms_mean": float(statistics.mean(mean_vals)),
                }
            )

    if repeats > 1:
        # Keep `results` stable/summary-focused for downstream parsing.
        results = [
            {
                "name": row["name"],
                "iters": int(args.iters),
                "warmup": int(args.warmup),
                "mean_ms": float(row["mean_ms_median"]),
                "p50_ms": float(row["p50_ms_median"]),
                "p95_ms": float(row["p95_ms_median"]),
            }
            for row in aggregate
        ]
    else:
        results = [asdict(r) for r in all_results[0]]

    payload = {
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
            "repeats": repeats,
            "vary_seed": bool(args.vary_seed),
            "pose_T": args.pose_T,
            "pose_J": args.pose_J,
            "videos": args.videos,
            "windows_per_video": args.windows_per_video,
            "stream_src_frames": args.stream_src_frames,
        },
        "results": results,
    }
    if aggregate:
        payload["aggregate"] = aggregate
        payload["per_repeat_results"] = [[asdict(r) for r in rep] for rep in all_results]

    print(json.dumps(payload, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
