#!/usr/bin/env python3
"""Temporal physical-time audit for window artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_gate_overrides(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[err] gates config not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[err] invalid gates json: {p}: {e}") from e
    sect = data.get("temporal", data)
    return sect if isinstance(sect, dict) else {}


def _window_shape_and_fps(fp: Path, fps_default: float) -> tuple[int, float]:
    with np.load(fp, allow_pickle=False) as z:
        joints = np.asarray(z["joints"] if "joints" in z.files else z["xy"], dtype=np.float32)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0]) if "fps" in z.files else float(fps_default)
    return int(joints.shape[0]), float(fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates_json", default="ops/configs/audit_gates.json")
    ap.add_argument("--processed_root", default="data/processed")
    ap.add_argument("--datasets", default="le2i,urfd,caucafall,muvim")
    ap.add_argument("--stride_frames", type=int, default=12)
    ap.add_argument("--fps_default", type=float, default=25.0)
    ap.add_argument("--max_windows_per_dataset", type=int, default=300)
    ap.add_argument("--target_window_seconds", type=float, default=1.92)
    ap.add_argument("--target_stride_seconds", type=float, default=0.48)
    ap.add_argument("--window_tol", type=float, default=0.15)
    ap.add_argument("--stride_tol", type=float, default=0.05)
    ap.add_argument("--strict_datasets", default="le2i,urfd")
    ap.add_argument("--fps_expect_le2i_urfd", type=float, default=25.0)
    ap.add_argument("--fps_tol", type=float, default=0.25)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    gate = _load_gate_overrides(str(args.gates_json).strip())
    if gate:
        args.target_window_seconds = float(gate.get("target_window_seconds", args.target_window_seconds))
        args.target_stride_seconds = float(gate.get("target_stride_seconds", args.target_stride_seconds))
        args.window_tol = float(gate.get("window_tol", args.window_tol))
        args.stride_tol = float(gate.get("stride_tol", args.stride_tol))
        args.strict_datasets = str(gate.get("strict_datasets", args.strict_datasets))
        args.fps_expect_le2i_urfd = float(gate.get("fps_expect_le2i_urfd", args.fps_expect_le2i_urfd))
        args.fps_tol = float(gate.get("fps_tol", args.fps_tol))
        args.max_windows_per_dataset = int(gate.get("max_windows_per_dataset", args.max_windows_per_dataset))

    root = Path(args.processed_root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    strict_datasets = {d.strip() for d in args.strict_datasets.split(",") if d.strip()}
    report: dict[str, Any] = {"datasets": {}, "failures": []}

    for ds in datasets:
        ds_root = root / ds
        eval_dirs = sorted(ds_root.glob("windows_eval_W*_S*"))
        if not eval_dirs:
            continue
        files = sorted(eval_dirs[-1].rglob("*.npz"))[: max(1, int(args.max_windows_per_dataset))]
        if not files:
            continue

        frame_counts: list[float] = []
        fps_vals: list[float] = []
        for fp in files:
            T, fps = _window_shape_and_fps(fp, args.fps_default)
            frame_counts.append(float(T))
            fps_vals.append(float(fps))

        mean_T = float(np.mean(frame_counts))
        mean_fps = float(np.mean(fps_vals))
        window_s = mean_T / max(mean_fps, 1e-6)
        stride_s = float(args.stride_frames) / max(mean_fps, 1e-6)
        report["datasets"][ds] = {
            "n_windows": len(files),
            "window_frames_mean": mean_T,
            "fps_effective_mean": mean_fps,
            "window_seconds": window_s,
            "stride_seconds": stride_s,
        }

        if ds in strict_datasets and abs(window_s - args.target_window_seconds) > args.window_tol:
            report["failures"].append(
                f"{ds}: window_seconds {window_s:.4f} outside target {args.target_window_seconds:.4f} +/- {args.window_tol:.4f}"
            )
        if ds in strict_datasets and abs(stride_s - args.target_stride_seconds) > args.stride_tol:
            report["failures"].append(
                f"{ds}: stride_seconds {stride_s:.4f} outside target {args.target_stride_seconds:.4f} +/- {args.stride_tol:.4f}"
            )
        if ds in strict_datasets and abs(mean_fps - args.fps_expect_le2i_urfd) > args.fps_tol:
            report["failures"].append(
                f"{ds}: fps {mean_fps:.4f} outside expected {args.fps_expect_le2i_urfd:.2f} +/- {args.fps_tol:.2f}"
            )

    out_json = args.out_json.strip() or "artifacts/reports/temporal_span_latest.json"
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] temporal report: {out}")

    if report["failures"]:
        print("[fail] temporal audit failures:")
        for f in report["failures"]:
            print(f" - {f}")
        raise SystemExit(1)
    print("[ok] temporal audit passed")


if __name__ == "__main__":
    main()
