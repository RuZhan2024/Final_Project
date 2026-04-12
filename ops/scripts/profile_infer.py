#!/usr/bin/env python3
"""Simple inference profiling for deploy readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from fall_detection.core.features import read_window_npz


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--profile", default="cpu_local")
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--fps_default", type=float, default=25.0)
    ap.add_argument("--io_only", type=int, default=1, help="1: profile window read path only (no torch/model).")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    files = sorted(Path(args.win_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"[ERR] no windows in {args.win_dir}")

    model = None
    feat_cfg = None
    two_stream = False
    arch = str(args.arch).lower()
    use_model = False

    if int(args.io_only) == 0 and args.ckpt.strip():
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_file():
            import torch  # lazy import for restricted environments
            from fall_detection.deploy.common import (
                build_input_from_raw,
                load_model_bundle,
                load_window_raw,
                predict_prob,
            )

            device = torch.device("cpu")
            model, arch_ckpt, _model_cfg, feat_cfg, two_stream, _fps = load_model_bundle(str(ckpt_path), device)
            arch = arch_ckpt
            use_model = True
        else:
            print(f"[warn] ckpt missing, profiling I/O-only path: {ckpt_path}")

    sample = files[0]
    if use_model:
        from fall_detection.deploy.common import build_input_from_raw, load_window_raw, predict_prob

        raw = load_window_raw(str(sample), fps_default=args.fps_default)

    # Warmup
    for _ in range(max(0, int(args.warmup))):
        if use_model and model is not None and feat_cfg is not None:
            X, _m = build_input_from_raw(raw, feat_cfg, arch, two_stream=two_stream)
            _ = predict_prob(model, arch, X, device=device, two_stream=two_stream)
        else:
            _ = read_window_npz(str(sample), fps_default=args.fps_default)

    lat_ms: list[float] = []
    runs = max(1, int(args.runs))
    for i in range(runs):
        fp = files[i % len(files)]
        t0 = time.perf_counter()
        if use_model:
            raw = load_window_raw(str(fp), fps_default=args.fps_default)
        else:
            _ = read_window_npz(str(fp), fps_default=args.fps_default)
        if use_model and model is not None and feat_cfg is not None:
            X, _m = build_input_from_raw(raw, feat_cfg, arch, two_stream=two_stream)
            _ = predict_prob(model, arch, X, device=device, two_stream=two_stream)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)

    report = {
        "profile": args.profile,
        "arch": arch,
        "with_model": bool(use_model and model is not None),
        "window_dir": str(args.win_dir),
        "n_windows_available": len(files),
        "warmup": int(args.warmup),
        "runs": runs,
        "latency_ms": {
            "mean": float(np.mean(lat_ms)),
            "median": _percentile(lat_ms, 50),
            "p95": _percentile(lat_ms, 95),
            "min": float(np.min(lat_ms)),
            "max": float(np.max(lat_ms)),
        },
    }

    out_json = args.out_json.strip()
    if not out_json:
        out_json = f"artifacts/reports/infer_profile_{args.profile}.json"
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] profile report: {out}")
    print(json.dumps(report["latency_ms"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
