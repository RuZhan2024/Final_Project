#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot F1 vs tau_high from sweep reports.

Supports:
- JSON reports (metrics.py style)
- YAML reports (fit_ops.py style)

Accepted sweep formats:
1) sweep as dict of arrays:
   sweep = {"thr":[...], "f1":[...]}
2) sweep as list of dicts:
   sweep = [{"thr":0.5,"f1":0.8,...}, ...]

Pareto markers:
- sweep_meta.pareto_idx: list[int]
- sweep_meta.pareto: list[int] OR boolean mask list[bool]
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml  # PyYAML
except Exception:
    yaml = None


def _load_report(path: str) -> dict:
    p = Path(path)
    suffix = p.suffix.lower()

    # Use utf-8 with ignore, since some envs embed non-utf8 in logs/paths.
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not available, cannot read .yaml/.yml reports.")
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            obj = yaml.safe_load(f)
            return obj if isinstance(obj, dict) else {"_root": obj}

    # default: json
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def _extract_thr_f1(rep: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (thr, f1, pareto_idx)
    """
    sweep = rep.get("sweep")

    thr: List[float] = []
    f1: List[float] = []

    # Format 1: dict-of-arrays
    if isinstance(sweep, dict):
        thr = list(sweep.get("thr") or sweep.get("tau_high") or [])
        f1 = list(sweep.get("f1") or sweep.get("event_f1") or [])
    # Format 2: list-of-dicts
    elif isinstance(sweep, list):
        for r in sweep:
            if not isinstance(r, dict):
                continue
            t = r.get("thr", r.get("tau_high", None))
            v = r.get("f1", r.get("event_f1", None))
            if t is None or v is None:
                continue
            thr.append(float(t))
            f1.append(float(v))

    thr_arr = np.asarray(thr, dtype=float)
    f1_arr = np.asarray(f1, dtype=float)

    # Pareto index extraction (supports idx list or bool mask)
    pareto_idx = np.array([], dtype=int)
    meta = rep.get("sweep_meta") or {}
    pareto = meta.get("pareto_idx", None)
    if pareto is None:
        pareto = meta.get("pareto", None)

    if pareto is not None and thr_arr.size > 0:
        try:
            arr = np.asarray(pareto)
            if arr.dtype == bool:
                # boolean mask
                idx = np.where(arr[: thr_arr.size])[0]
            else:
                # assume indices
                idx = np.asarray(arr, dtype=int)
                idx = idx[(idx >= 0) & (idx < thr_arr.size)]
            pareto_idx = idx.astype(int)
        except Exception:
            pareto_idx = np.array([], dtype=int)

    return thr_arr, f1_arr, pareto_idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot F1 vs tau_high from sweep reports (JSON or YAML).")
    ap.add_argument("--reports", nargs="+", required=True, help="One or more metrics JSON / fit_ops YAML files")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels (defaults to file stems)")
    ap.add_argument("--out_fig", required=True, help="Output PNG")
    ap.add_argument("--show_pareto", type=int, default=1, help="If 1, mark pareto sweep points when available")
    args = ap.parse_args()

    labels = args.labels if args.labels else [Path(r).stem for r in args.reports]
    if len(labels) != len(args.reports):
        ap.error("--labels must have the same length as --reports")

    plt.figure()

    for report_path, label in zip(args.reports, labels):
        rep = _load_report(report_path)
        thr, f1, pareto_idx = _extract_thr_f1(rep)

        if thr.size == 0 or f1.size == 0:
            print(f"[warn] missing sweep thr/f1 in {report_path}")
            continue

        # sort by threshold for a clean curve
        order = np.argsort(thr)
        thr = thr[order]
        f1 = f1[order]

        plt.plot(thr, f1, marker="o", linestyle="-", label=str(label))

        if int(args.show_pareto) == 1 and pareto_idx.size > 0:
            # If pareto indices were from pre-sort order, map them.
            # Best-effort: assume pareto idx refer to the unsorted arrays, so convert.
            try:
                inv = np.empty_like(order)
                inv[order] = np.arange(order.size)
                idx_sorted = inv[pareto_idx]
                idx_sorted = idx_sorted[(idx_sorted >= 0) & (idx_sorted < thr.size)]
                if idx_sorted.size > 0:
                    plt.scatter(thr[idx_sorted], f1[idx_sorted], s=40, label=f"{label} pareto")
            except Exception:
                pass

    plt.xlabel("tau_high (threshold)")
    plt.ylabel("Event F1")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[done] wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
