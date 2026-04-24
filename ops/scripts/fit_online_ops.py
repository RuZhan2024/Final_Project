#!/usr/bin/env python3
"""Fit online operating points by simulating the monitor-state tracker.

Unlike offline threshold sweeps over raw probabilities, this script replays the
same low-motion and structural-quality suppressions that affect the live monitor
path, then chooses OP candidates from the resulting video-level behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from applications.backend.deploy_runtime import get_specs, predict_spec
from applications.backend.online_alert import OnlineAlertTracker
from applications.backend.routes.monitor import (
    _DEFAULT_LIVE_GUARD_BY_DATASET,
    _LOW_MOTION_MEMORY_WINDOWS,
    _direct_window_stats,
    _effective_target_fps,
    _window_motion_score,
    _window_quality_block,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict-shaped config object."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _video_id_for(npz_path: Path, dataset_code: str) -> str:
    """Recover the source-video grouping key used for one window NPZ filename."""
    stem = npz_path.stem
    if "__w" in stem:
        stem = stem.split("__w", 1)[0]
    if dataset_code == "caucafall":
        parts = stem.split("__")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1].replace('_', ' ')}"
    return stem


def _group_videos(test_dir: Path, dataset_code: str) -> Dict[str, List[Path]]:
    """Group replay windows by source video so online alerts are simulated per stream."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(test_dir.glob("*.npz")):
        groups[_video_id_for(f, dataset_code)].append(f)
    return groups


def _candidate_cfg(base_alert_cfg: Dict[str, Any], tau_high: float, tau_low_ratio: float, ema_alpha: float, k: int, n: int) -> Dict[str, Any]:
    """Build one candidate alert-policy config from a shared base policy."""
    tau_low = max(0.0, min(tau_high - 1e-6, tau_high * tau_low_ratio))
    cfg = dict(base_alert_cfg)
    cfg["tau_high"] = float(tau_high)
    cfg["tau_low"] = float(tau_low)
    cfg["ema_alpha"] = float(ema_alpha)
    cfg["k"] = int(k)
    cfg["n"] = int(n)
    return cfg


def _precompute_videos(*, spec_key: str, dataset_code: str, test_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Precompute per-window diagnostics shared by every candidate alert policy."""
    spec = get_specs()[spec_key]
    groups = _group_videos(test_dir, dataset_code)
    min_motion = float(_DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {}).get("min_motion_for_fall", 0.020))
    expected_fps_default = 23.0 if dataset_code == "caucafall" else 25.0
    out: Dict[str, List[Dict[str, Any]]] = {}
    for video_id, files in groups.items():
        rows: List[Dict[str, Any]] = []
        for idx, f in enumerate(files):
            x = np.load(f, allow_pickle=True)
            xy = x["joints"].tolist()
            conf = x["conf"].tolist()
            fps = float(np.asarray(x["fps"]).item())
            expected_fps = expected_fps_default
            effective_fps = _effective_target_fps(expected_fps=expected_fps, raw_fps_est=fps)
            raw_stats = _direct_window_stats(xy, conf, effective_fps=effective_fps)
            qdiag = _window_quality_block(
                raw_stats=raw_stats,
                expected_fps=expected_fps,
                effective_fps=effective_fps,
                target_T=48,
                dataset_code=dataset_code,
                min_fps_ratio_override=float(_DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {}).get("min_fps_ratio", 0.70)),
                min_frames_ratio_override=0.60,
                min_coverage_ratio_override=0.85,
            )
            structural_quality_block = bool(qdiag.get("low_frames", False) or qdiag.get("low_coverage", False))
            mu_out = predict_spec(
                spec_key=spec_key,
                joints_xy=xy,
                conf=conf,
                fps=float(expected_fps),
                target_T=48,
                op_code="OP-2",
                use_mc=False,
            )
            motion_score = _window_motion_score(xy)
            label = int(np.asarray(x["label"]).item())
            overlap_frac = float(np.asarray(x["overlap_frac"]).item()) if "overlap_frac" in x.files else 0.0
            gt = 1 if (label == 1 or overlap_frac > 0.0) else 0
            t_s = float(((idx + 1) * 12) / max(1e-6, fps))
            rows.append(
                {
                    "file": f.name,
                    "gt": gt,
                    "mu": float(mu_out.get("mu") or mu_out.get("p_det") or 0.0),
                    "motion_score": None if motion_score is None else float(motion_score),
                    "low_motion_block": bool(motion_score is not None and float(motion_score) < min_motion),
                    "structural_quality_block": structural_quality_block,
                    "t_s": t_s,
                }
            )
        out[video_id] = rows
    return out


def _simulate_video(rows: List[Dict[str, Any]], cfg: Dict[str, Any], min_motion: float, memory_windows: int) -> int:
    """Replay one video's window stream through the online alert tracker."""
    tracker = OnlineAlertTracker(cfg)
    motion_hist: List[float] = []
    for row in rows:
        motion_score = row["motion_score"]
        if motion_score is not None and math.isfinite(float(motion_score)):
            motion_hist.append(float(motion_score))
        if len(motion_hist) > memory_windows:
            motion_hist = motion_hist[-memory_windows:]
        recent_motion_support = any(float(v) >= float(min_motion) for v in motion_hist)
        tau_low = float(cfg["tau_low"])
        p_in = float(row["mu"])
        if (
            bool(row["low_motion_block"])
            and not recent_motion_support
        ) or bool(row["structural_quality_block"]):
            # Force blocked windows below tau_low so the tracker sees the same
            # downgrade shape as the runtime path instead of a mere annotation.
            p_in = min(float(p_in), float(tau_low) - 0.02)
        res = tracker.step(p=p_in, t_s=float(row["t_s"]))
        if res.triage_state == "fall":
            return 1
    return 0


def _score_summary(videos: Dict[str, List[Dict[str, Any]]], cfg: Dict[str, Any], dataset_code: str) -> Dict[str, Any]:
    """Score one candidate config at the source-video level."""
    min_motion = float(_DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {}).get("min_motion_for_fall", 0.020))
    tp = tn = fp = fn = 0
    rows: List[Dict[str, Any]] = []
    for video_id, seq in videos.items():
        gt = max(int(r["gt"]) for r in seq)
        pred = _simulate_video(seq, cfg, min_motion=min_motion, memory_windows=_LOW_MOTION_MEMORY_WINDOWS)
        if gt and pred:
            tp += 1
            status = "TP"
        elif (not gt) and (not pred):
            tn += 1
            status = "TN"
        elif pred:
            fp += 1
            status = "FP"
        else:
            fn += 1
            status = "FN"
        rows.append({"video_id": video_id, "gt": gt, "pred": pred, "status": status})
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2 * precision * recall) / max(1e-9, precision + recall)) if (tp > 0) else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": rows,
    }


def _pick_ops(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pick OP1/OP2/OP3 from the simulated search table.

    The three selectors intentionally optimize different tradeoffs rather than
    returning three nearest thresholds from one scalar metric.
    """
    def _key_high_tau(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (float(r["recall"]), -float(r["FP"]), float(r["f1"]), float(r["tau_high"]))

    def _key_balanced(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (-float(r["FP"]), float(r["f1"]), float(r["recall"]), float(r["tau_high"]))

    def _key_safe(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (-float(r["FP"]), float(r["recall"]), float(r["f1"]), float(r["tau_high"]))

    op1 = max(results, key=_key_high_tau)
    zero_fp = [r for r in results if int(r["FP"]) == 0]
    if zero_fp:
        op2 = max(zero_fp, key=_key_balanced)
        op3 = max(zero_fp, key=_key_safe)
    else:
        op2 = max(results, key=_key_balanced)
        op3 = max(results, key=_key_safe)
    return {"OP1": op1, "OP2": op2, "OP3": op3}


def main() -> None:
    """Search alert-policy candidates and persist the selected operating points."""
    ap = argparse.ArgumentParser(description="Fit online operating points using the monitor-state simulation.")
    ap.add_argument("--spec_key", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--tau_min", type=float, default=0.20)
    ap.add_argument("--tau_max", type=float, default=0.80)
    ap.add_argument("--tau_step", type=float, default=0.02)
    ap.add_argument("--tau_low_ratio", type=float, default=0.78)
    ap.add_argument("--ema", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument("--k_values", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--n_values", nargs="+", type=int, default=[2, 3, 4])
    args = ap.parse_args()

    spec = get_specs()[args.spec_key]
    dataset_code = str(spec.dataset)
    test_dir = Path(args.test_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    videos = _precompute_videos(spec_key=args.spec_key, dataset_code=dataset_code, test_dir=test_dir)
    base_alert_cfg = dict(spec.alert_cfg or {})

    results: List[Dict[str, Any]] = []
    tau = float(args.tau_min)
    # Sweep tau_high plus persistence settings because online operating points
    # depend on tracker memory, not just a single scalar threshold.
    while tau <= float(args.tau_max) + 1e-9:
        for ema_alpha in args.ema:
            for n in args.n_values:
                for k in args.k_values:
                    if k > n:
                        continue
                    cfg = _candidate_cfg(
                        base_alert_cfg=base_alert_cfg,
                        tau_high=float(tau),
                        tau_low_ratio=float(args.tau_low_ratio),
                        ema_alpha=float(ema_alpha),
                        k=int(k),
                        n=int(n),
                    )
                    score = _score_summary(videos, cfg, dataset_code=dataset_code)
                    results.append(
                        {
                            "tau_high": float(cfg["tau_high"]),
                            "tau_low": float(cfg["tau_low"]),
                            "ema_alpha": float(cfg["ema_alpha"]),
                            "k": int(cfg["k"]),
                            "n": int(cfg["n"]),
                            **{k: v for k, v in score.items() if k != "rows"},
                        }
                    )
        tau += float(args.tau_step)

    picked = _pick_ops(results)
    out = {
        "spec_key": args.spec_key,
        "dataset_code": dataset_code,
        "test_dir": str(test_dir),
        "search": {
            "tau_min": float(args.tau_min),
            "tau_max": float(args.tau_max),
            "tau_step": float(args.tau_step),
            "tau_low_ratio": float(args.tau_low_ratio),
            "ema": [float(x) for x in args.ema],
            "k_values": [int(x) for x in args.k_values],
            "n_values": [int(x) for x in args.n_values],
            "memory_windows": int(_LOW_MOTION_MEMORY_WINDOWS),
        },
        "picked": picked,
        "results": results,
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"spec_key": args.spec_key, "picked": picked}, indent=2))


if __name__ == "__main__":
    main()
