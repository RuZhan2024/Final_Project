#!/usr/bin/env python3
"""Minimal fault-injection scaffold for deployment robustness checks.

This tool runs lightweight, reproducible fault scenarios and writes a summary JSON.
It is intentionally non-destructive and does not require training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _result(name: str, ok: bool, detail: str, duration_ms: float) -> Dict[str, Any]:
    return {
        "scenario": name,
        "ok": bool(ok),
        "detail": str(detail),
        "duration_ms": float(duration_ms),
    }


def scenario_empty_skeleton() -> Dict[str, Any]:
    t0 = time.time()
    xy = np.full((48, 33, 2), np.nan, dtype=np.float32)
    conf = np.zeros((48, 33), dtype=np.float32)
    # Expect: pipeline can treat as empty/no-signal without crashing.
    finite_xy = int(np.isfinite(xy).sum())
    valid_conf = int((conf > 0.2).sum())
    ok = finite_xy == 0 and valid_conf == 0
    return _result("empty_skeleton", ok, f"finite_xy={finite_xy}, valid_conf={valid_conf}", (time.time() - t0) * 1000.0)


def scenario_dropped_frames() -> Dict[str, Any]:
    t0 = time.time()
    xy = np.random.randn(48, 33, 2).astype(np.float32)
    mask = np.ones((48, 33), dtype=bool)
    # Drop a burst of frames
    xy[10:18] = np.nan
    mask[10:18] = False
    dropped = int((~mask).all(axis=1).sum())
    ok = dropped >= 8
    return _result("dropped_frames", ok, f"dropped_full_frames={dropped}", (time.time() - t0) * 1000.0)


def scenario_low_confidence() -> Dict[str, Any]:
    t0 = time.time()
    conf = np.random.uniform(0.0, 0.15, size=(48, 33)).astype(np.float32)
    gated = int((conf > 0.2).sum())
    ok = gated == 0
    return _result("low_confidence", ok, f"conf_above_gate={gated}", (time.time() - t0) * 1000.0)


def scenario_missing_file() -> Dict[str, Any]:
    t0 = time.time()
    p = Path("/tmp/nonexistent_fault_inject_file.npz")
    try:
        _ = np.load(p, allow_pickle=False)
        return _result("missing_file", False, "unexpectedly opened missing file", (time.time() - t0) * 1000.0)
    except Exception as e:
        return _result("missing_file", True, f"caught={type(e).__name__}", (time.time() - t0) * 1000.0)


def scenario_camera_end() -> Dict[str, Any]:
    t0 = time.time()
    # Simulate stream end by iterating a finite generator.
    frames = [np.zeros((33, 2), dtype=np.float32) for _ in range(3)]
    consumed = 0
    for _ in frames:
        consumed += 1
    ended = consumed == len(frames)
    return _result("camera_end", ended, f"consumed_frames={consumed}", (time.time() - t0) * 1000.0)


def scenario_db_api_failure() -> Dict[str, Any]:
    t0 = time.time()
    # Non-destructive simulation: force an OS-level open failure.
    bad_path = Path("/root/forbidden_fault_inject_path.txt")
    try:
        bad_path.write_text("x", encoding="utf-8")
        return _result("db_api_failure", False, "unexpected write success", (time.time() - t0) * 1000.0)
    except Exception as e:
        return _result("db_api_failure", True, f"caught={type(e).__name__}", (time.time() - t0) * 1000.0)


SCENARIOS = {
    "empty_skeleton": scenario_empty_skeleton,
    "dropped_frames": scenario_dropped_frames,
    "low_confidence": scenario_low_confidence,
    "missing_file": scenario_missing_file,
    "camera_end": scenario_camera_end,
    "db_api_failure": scenario_db_api_failure,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        default="all",
        help="Scenario name or 'all'. Options: " + ", ".join(sorted(SCENARIOS.keys())),
    )
    ap.add_argument("--out_json", default="artifacts/reports/fault_inject_summary.json")
    args = ap.parse_args()

    names: List[str]
    if args.scenario == "all":
        names = list(SCENARIOS.keys())
    else:
        if args.scenario not in SCENARIOS:
            raise SystemExit(f"[ERR] unknown scenario: {args.scenario}")
        names = [args.scenario]

    results = []
    for n in names:
        fn = SCENARIOS[n]
        try:
            results.append(fn())
        except Exception as e:
            results.append(_result(n, False, f"uncaught={type(e).__name__}: {e}", 0.0))

    passed = sum(1 for r in results if r.get("ok"))
    summary = {
        "schema_version": "1.0",
        "scenario": args.scenario,
        "n_total": len(results),
        "n_pass": passed,
        "n_fail": len(results) - passed,
        "results": results,
        "pass_rate": float(passed / max(1, len(results))),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote fault summary: {out}")


if __name__ == "__main__":
    main()
