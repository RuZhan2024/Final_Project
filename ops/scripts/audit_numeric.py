#!/usr/bin/env python3
"""Cross-dataset numeric fingerprint audit for window artifacts."""

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
    sect = data.get("numeric", data)
    return sect if isinstance(sect, dict) else {}


def _stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p1": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def _merge_stats(seq: list[dict[str, float]]) -> dict[str, float]:
    if not seq:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p1": 0.0, "p99": 0.0}
    out: dict[str, float] = {}
    for k in seq[0].keys():
        vals = np.asarray([s[k] for s in seq], dtype=np.float32)
        if k in {"min", "p1"}:
            out[k] = float(np.min(vals))
        elif k in {"max", "p99"}:
            out[k] = float(np.max(vals))
        else:
            out[k] = float(np.mean(vals))
    return out


def _read_window(fp: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(fp, allow_pickle=False) as z:
        joints = np.asarray(z["joints"] if "joints" in z.files else z["xy"], dtype=np.float32)
        motion = np.asarray(z["motion"], dtype=np.float32) if "motion" in z.files else None
        conf = np.asarray(z["conf"], dtype=np.float32) if "conf" in z.files else None
    return joints, motion, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates_json", default="configs/audit_gates.json")
    ap.add_argument("--processed_root", default="data/processed")
    ap.add_argument("--datasets", default="le2i,urfd,caucafall,muvim")
    ap.add_argument("--max_windows_per_dataset", type=int, default=300)
    ap.add_argument("--ref_dataset", default="le2i")
    ap.add_argument("--std_ratio_min", type=float, default=0.5)
    ap.add_argument("--std_ratio_max", type=float, default=2.0)
    ap.add_argument("--mean_delta_max", type=float, default=1.0)
    ap.add_argument("--abs_p99_max", type=float, default=5.0)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    gate = _load_gate_overrides(str(args.gates_json).strip())
    if gate:
        args.ref_dataset = str(gate.get("ref_dataset", args.ref_dataset))
        args.std_ratio_min = float(gate.get("std_ratio_min", args.std_ratio_min))
        args.std_ratio_max = float(gate.get("std_ratio_max", args.std_ratio_max))
        args.mean_delta_max = float(gate.get("mean_delta_max", args.mean_delta_max))
        args.abs_p99_max = float(gate.get("abs_p99_max", args.abs_p99_max))
        args.max_windows_per_dataset = int(gate.get("max_windows_per_dataset", args.max_windows_per_dataset))

    root = Path(args.processed_root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    report: dict[str, Any] = {"datasets": {}, "checks": {}, "failures": []}
    for ds in datasets:
        ds_root = root / ds
        eval_dirs = sorted(ds_root.glob("windows_eval_W*_S*"))
        if not eval_dirs:
            continue
        files = sorted(eval_dirs[-1].rglob("*.npz"))
        if not files:
            continue
        files = files[: max(1, int(args.max_windows_per_dataset))]

        joints_stats: list[dict[str, float]] = []
        motion_stats: list[dict[str, float]] = []
        conf_stats: list[dict[str, float]] = []
        for fp in files:
            joints, motion, conf = _read_window(fp)
            joints_stats.append(_stats(joints))
            if motion is not None:
                motion_stats.append(_stats(motion))
            if conf is not None:
                conf_stats.append(_stats(conf))

        report["datasets"][ds] = {
            "n_windows": len(files),
            "joints": _merge_stats(joints_stats),
            "motion": _merge_stats(motion_stats),
            "conf": _merge_stats(conf_stats),
        }

    ref = report["datasets"].get(args.ref_dataset)
    if ref is None:
        report["failures"].append(f"missing reference dataset stats: {args.ref_dataset}")
    else:
        ref_std = max(float(ref["joints"]["std"]), 1e-6)
        ref_mean = float(ref["joints"]["mean"])
        for ds, dsr in report["datasets"].items():
            std_ratio = float(dsr["joints"]["std"]) / ref_std
            mean_delta = abs(float(dsr["joints"]["mean"]) - ref_mean)
            abs_p99 = abs(float(dsr["joints"]["p99"]))
            report["checks"][ds] = {
                "std_ratio_vs_ref": std_ratio,
                "mean_delta_vs_ref": mean_delta,
                "abs_p99_joints": abs_p99,
            }
            if not (args.std_ratio_min <= std_ratio <= args.std_ratio_max):
                report["failures"].append(
                    f"{ds}: std_ratio {std_ratio:.4f} outside [{args.std_ratio_min}, {args.std_ratio_max}]"
                )
            if mean_delta > args.mean_delta_max:
                report["failures"].append(
                    f"{ds}: mean_delta {mean_delta:.4f} > {args.mean_delta_max}"
                )
            if abs_p99 > args.abs_p99_max:
                report["failures"].append(
                    f"{ds}: abs_p99 {abs_p99:.4f} > {args.abs_p99_max}"
                )

    out_json = args.out_json.strip()
    if not out_json:
        out_json = "artifacts/reports/numeric_fingerprint_latest.json"
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] numeric report: {out}")

    if report["failures"]:
        print("[fail] numeric audit failures:")
        for f in report["failures"]:
            print(f" - {f}")
        raise SystemExit(1)
    print("[ok] numeric audit passed")


if __name__ == "__main__":
    main()
